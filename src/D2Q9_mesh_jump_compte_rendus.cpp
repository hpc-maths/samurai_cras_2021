// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We repeat the test case from
// Grid refinement for aeroacoustics in the lattice Boltzmann method: A directional splitting approach

// We use the MRT from Lallemand

#include <math.h>
#include <vector>

#include <cxxopts.hpp>

#include <samurai/mr/adapt.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/hdf5.hpp>

#include "prediction_map_2d.hpp"
#include "boundary_conditions.hpp"


#include "utils_lbm_mr_2d.hpp"
#include "laguerre_interpolation.hpp"

#include <cmath>



double exact_solution(const double x, const double y, const double t, const double c0, const bool fine = false)
{
    // Parameters from the paper
    const double epsilon = 1.e-3;
    const double b = 1.e-1;
    const double alpha = std::log(2.) / (b*b);

    const double radius = std::sqrt(x*x + y*y);

    auto integrand = [radius, t, c0, epsilon, alpha] (double s)
    {
        return std::exp(s*(1 - s/(4*alpha))) * std::cos(c0*t*s) * std::cyl_bessel_j(0, radius*s);
    };

    double result = 0.;
    if (!fine)  {
        for (std::size_t idx = 0; idx < laguerre_points_20.size(); ++idx)  {
            result += laguerre_weights_20[idx] * integrand(laguerre_points_20[idx]);
        }
    }
    else
    {
        for (std::size_t idx = 0; idx < laguerre_points_100.size(); ++idx)  {
            result += laguerre_weights_100[idx] * integrand(laguerre_points_100[idx]);
        }
    }

    return epsilon/(2*alpha)*result;
}

template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    auto rotation_of_pi_over_two = [] (int alpha, int k, int h)
    {
        // Returns the rotation of (k, h) of an angle alpha * pi / 2.
        // All the operations are performed on integer, to be exact
        int cosinus = static_cast<int>(std::round(std::cos(alpha * M_PI / 2.)));
        int sinus   = static_cast<int>(std::round(std::sin(alpha * M_PI / 2.)));

        return std::pair<int, int> (cosinus * k - sinus   * h,
                                      sinus * k + cosinus * h);
    };

    // Transforms the coordinates to apply the rotation
    auto tau = [] (int delta, int k)
    {
        // The case in which delta = 0 is rather exceptional
        if (delta == 0) {
            return k;
        }
        else {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < tmp) ? (k - tmp) : (k - tmp + 1));
        }
    };

    auto tau_inverse = [] (int delta, int k)
    {
        if (delta == 0) {
            return k;
        }
        else
        {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < 0) ? (k + tmp) : (k + tmp - 1));
        }
    };

    for(std::size_t k = 0; k < max_level - min_level + 1; ++k)
    {
        int size = (1<<k);

        // We have 9 velocity out of which 8 are moving
        // 4 are moving along the axis, thus needing only 2 fluxes each (entering-exiting)
        // and 4 along the diagonals, thus needing  6 fluxes

        // 4 * 2 + 4 * 6 = 8 + 24 = 32
        data[k].resize(32);

        // Parallel velocities
        for (int alpha = 0; alpha <= 3; ++alpha)
        {
            for (int l = 0; l < size; ++l)
            {
                // The reference direction from which the other ones are computed is that of (1, 0)
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i   * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1)* size - 1), tau(k, j * size + l));

                data[k][0 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_in.first ), tau_inverse(k, rotated_in.second ));
                data[k][1 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));
            }
        }

        // Diagonal velocities

        // Translation of the indices from which we start saving the new computations
        int offset = 4 * 2;
        for (int alpha = 0; alpha <= 3; ++alpha)
        {

            // First side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i   * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1)* size - 1), tau(k, j * size + l));

                data[k][offset + 6 * alpha + 0] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 3] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));

            }
            // Cell on the diagonal
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k,  i    * size - 1), tau(k,  j    * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i+1) * size - 1), tau(k, (j+1) * size - 1));

                data[k][offset + 6 * alpha + 1] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 4] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));

            }
            // Second side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha,  tau(k, i*size + l), tau(k,  j    * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha,  tau(k, i*size + l), tau(k, (j+1) * size - 1));

                data[k][offset + 6 * alpha + 2] += prediction(k, tau_inverse(k, rotated_in.first ),  tau_inverse(k, rotated_in.second ));
                data[k][offset + 6 * alpha + 5] += prediction(k, tau_inverse(k, rotated_out.first),  tau_inverse(k, rotated_out.second));
            }
        }
    }
    return data;
}

template<class Field>
void init_fields(Field & f, const double lambda = 1.)
{
    // Parameters from the paper
    const double epsilon = 1.e-3;
    const double b = 1.e-1;
    const double alpha = std::log(2.) / (b*b);

    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    auto mesh = f.mesh();
    f.fill(0);

    const double r1 = 1.0 / lambda;
    const double r2 = 1.0 / (lambda*lambda);
    const double r3 = 1.0 / (lambda*lambda*lambda);
    const double r4 = 1.0 / (lambda*lambda*lambda*lambda);
    const double cs2 = (lambda*lambda)/ 3.0; // Sound velocity of the lattice squared

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        auto radiussq = x*x + y*y;

        double rho = 1 + epsilon * std::exp(-alpha*radiussq);
        double qx = 0.;
        double qy = 0.;

        double m0 = rho;
        double m1 = qx;
        double m2 = qy;
        double m3 = -2*lambda*lambda*rho + 3./rho*(qx*qx + qy*qy);
        double m4 = -lambda*lambda*qx;
        double m5 = -lambda*lambda*qy;
        double m6 = lambda*lambda*lambda*lambda*rho - 3.*lambda*lambda/rho*(qx*qx + qy*qy);
        double m7 = (qx*qx-qy*qy)/rho;
        double m8 = qx*qy/rho;

        f[cell][0] = (1./9)*m0                                  -  (1./9)*r2*m3                                +   (1./9)*r4*m6                         ;
        f[cell][1] = (1./9)*m0   + (1./6)*r1*m1                 - (1./36)*r2*m3 - (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ;
        f[cell][2] = (1./9)*m0                  +  (1./6)*r1*m2 - (1./36)*r2*m3                -  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ;
        f[cell][3] = (1./9)*m0   - (1./6)*r1*m1                 - (1./36)*r2*m3 + (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ;
        f[cell][4] = (1./9)*m0                  -  (1./6)*r1*m2 - (1./36)*r2*m3                +  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ;
        f[cell][5] = (1./9)*m0   + (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ;
        f[cell][6] = (1./9)*m0   - (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ;
        f[cell][7] = (1./9)*m0   - (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ;
        f[cell][8] = (1./9)*m0   + (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ;
    });
}

template<class Field>
void prepare_ghosts(Field & f)
{
    constexpr std::size_t nvel = Field::size;
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh = f.mesh();
    const auto min_level = mesh.min_level();
    const auto max_level = mesh.max_level(); // We assume that only one level of difference exists
    const auto dl = max_level - min_level;

    // We first update the coarse cells beneath the refined ones
    auto to_project = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][min_level],
                                            mesh[mesh_id_t::cells][max_level]).on(min_level);
    to_project([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        // std::cout<<"k = "<<k<<"  h = "<<h<<std::endl;

        for (int k_cell = k.start; k_cell < k.end; ++k_cell)    {
            const int k_fine_start =  k_cell      * (1<<dl);
            const int k_fine_end   = (k_cell + 1) * (1<<dl);

            for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {
                f(field_num, min_level, {k_cell, k_cell + 1}, h) =  xt::mean(.5*(f(field_num, max_level, {k_fine_start, k_fine_end}, 2*h) +
                                                                               f(field_num, max_level, {k_fine_start, k_fine_end}, 2*h+1)));
            }
        }
    });
    // We eventually update the cells at the finest level above the coarse ones
    // Once more, we assume that there is only a one-level jump
    auto to_predict = samurai::intersection(samurai::difference(mesh[mesh_id_t::cells_and_ghosts][max_level],
                                                                mesh[mesh_id_t::cells][max_level]),
                                            mesh.domain());
    to_predict([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        int parity_y = ((h & 1) == 1) ? -1 : 1;
        auto father_y = h >> 1;

        for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {
            for (int k_cell = k.start; k_cell < k.end; ++k_cell)    {
                auto father_x = k_cell >> 1;
                interval_t father_int {father_x, father_x+1};
                int parity_x = ((k_cell & 1) == 1) ? -1 : 1;

                    // // With the Haar wavelet
                    // f(field_num, max_level, {k_cell, k_cell + 1}, h) = f(field_num, min_level, father_int, father_y) ;
                    f(field_num, max_level, {k_cell, k_cell + 1}, h) = f(field_num, min_level, father_int, father_y)
                    + 1./8*parity_x*(f(field_num, min_level, father_int - 1, father_y) - f(field_num, min_level, father_int + 1, father_y))
                    + 1./8*parity_y*(f(field_num, min_level, father_int, father_y - 1) - f(field_num, min_level, father_int, father_y + 1))
                    - 1./64*parity_x*parity_y*(f(field_num, min_level, father_int+1, father_y + 1)+f(field_num, min_level, father_int-1, father_y - 1)
                                              -f(field_num, min_level, father_int+1, father_y - 1)-f(field_num, min_level, father_int-1, father_y + 1)); // Check but ok
            }
        }
    });
}

// I cannot use the previous one
// to deal with scalar fields ... pity
template<class Field>
void prepare_ghosts_one_field(Field & f)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh = f.mesh();
    const auto min_level = mesh.min_level();
    const auto max_level = mesh.max_level(); // We assume that only one level of difference exists
    const auto dl = max_level - min_level;

    // We eventually update the cells at the finest level above the coarse ones
    // Once more, we assume that there is only a one-level jump
    auto to_predict = samurai::intersection(samurai::difference(mesh[mesh_id_t::cells_and_ghosts][max_level],
                                                                mesh[mesh_id_t::cells][max_level]),
                                            mesh.domain());
    to_predict([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        int parity_y = ((h & 1) == 1) ? -1 : 1;
        auto father_y = h >> 1;

        for (int k_cell = k.start; k_cell < k.end; ++k_cell)    {
            auto father_x = k_cell >> 1;
            interval_t father_int {father_x, father_x+1};
            int parity_x = ((k_cell & 1) == 1) ? -1 : 1;

                // // With the Haar wavelet
                // f(field_num, max_level, {k_cell, k_cell + 1}, h) = f(field_num, min_level, father_int, father_y) ;
                f(max_level, {k_cell, k_cell + 1}, h) = f(min_level, father_int, father_y)
                + 1./8*parity_x*(f( min_level, father_int - 1, father_y) - f( min_level, father_int + 1, father_y))
                + 1./8*parity_y*(f( min_level, father_int, father_y - 1) - f( min_level, father_int, father_y + 1))
                - 1./64*parity_x*parity_y*(f(min_level, father_int+1, father_y + 1)+f(min_level, father_int-1, father_y - 1)
                                          -f(min_level, father_int+1, father_y - 1)-f(min_level, father_int-1, father_y + 1)); // Check but ok
        }
    });
}

template<class Field, class Func, class PredCoeff>
void one_time_step(Field & f, Func && update_bc_for_level, const PredCoeff & pred_coeff,
                 const double lambda, const double mu, const double zeta, std::size_t ite)
{
    constexpr std::size_t nvel = Field::size;
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh = f.mesh();
    const auto min_level = mesh.min_level();
    const auto max_level = mesh.max_level();
    const auto dl = max_level - min_level;

    auto advected = samurai::make_field<double, nvel>("advected_f", mesh);
    advected.fill(0.);

    Field fluxes{"fluxes", mesh};
    fluxes.array().fill(0.);

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));
    prepare_ghosts(f);

    // std::stringstream str;
    // str << "D2Q9_mesh_jump_compte_rendus_debug_ite-" << ite;

    // auto rho = samurai::make_field<double, 1>("rho", mesh);

    // samurai::for_each_cell(mesh[mesh_id_t::all_cells], [&](auto &cell) {
    //     rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4]
    //                            + f[cell][5] + f[cell][6] + f[cell][7] + f[cell][8];

    // });
    // samurai::save(str.str().data(), {true, true}, mesh, rho);

    double l1 = lambda;
    double l2 = l1 * lambda;
    double l3 = l2 * lambda;
    double l4 = l3 * lambda;

    double r1 = 1.0 / lambda;
    double r2 = 1.0 / (lambda*lambda);
    double r3 = 1.0 / (lambda*lambda*lambda);
    double r4 = 1.0 / (lambda*lambda*lambda*lambda);

    // Stream
    for (std::size_t level = 0; level <= max_level; ++level)    {

        const std::size_t dl = max_level - level;
        const double coeff = 1. / (1 << (2*dl));

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        leaves([&](auto &interval, auto& index) {
            auto k = interval; // Logical index in x
            auto h = index[0];    // Logical index in y

            std::array<int, 16> flx_num {1, 3, 5, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 29, 30, 31};
            std::array<int, 16> flx_vel {1, 2, 3, 4, 5,  5,  5,  6,  6,  6,  7,  7,  7,  8,  8,  8};

            for (int idx = 0; idx < flx_num.size(); ++idx)  {
                for(auto &c: pred_coeff[dl][flx_num[idx]].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;

                    // Be careful about the - sign because we are dealing with exiting fluxes
                    fluxes(flx_vel[idx], level, k, h) -= coeff * c.second * f(flx_vel[idx], level, k + stencil_x, h + stencil_y);
                }
            }
        });
        leaves([&](auto &interval, auto& index) {
            auto k = interval; // Logical index in x
            auto h = index[0];    // Logical index in y

            std::array<int, 16> flx_num {0, 2, 4, 6, 8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28};
            std::array<int, 16> flx_vel {1, 2, 3, 4, 5, 5, 5,  6,  6,  6,  7,  7,  7,  8,  8,  8};

            for (int idx = 0; idx < flx_num.size(); ++idx)  {
                for(auto &c: pred_coeff[dl][flx_num[idx]].coeff)
                {
                    coord_index_t stencil_x, stencil_y;
                    std::tie(stencil_x, stencil_y) = c.first;

                    fluxes(flx_vel[idx], level, k, h) += coeff * c.second * f(flx_vel[idx], level, k + stencil_x, h + stencil_y);
                }
            }
        });
        leaves([&](auto &interval, auto& index) {
            auto k = interval; // Logical index in x
            auto h = index[0];    // Logical index in y

            advected(0, level, k, h) = f(0, level, k, h); // Not moving so no flux
            for (int pop = 1; pop < 9; ++pop)  {
                advected(pop, level, k, h) = f(pop, level, k, h) + fluxes(pop, level, k, h);
            }
        });
    }
    for (std::size_t level = 0; level <= max_level; ++level)    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        leaves([&](auto &interval, auto& index) {
            auto k = interval; // Logical index in x
            auto h = index[0];    // Logical index in y

            auto m0 = xt::eval(       advected(0, level, k, h) +   advected(1, level, k, h) +   advected(2, level, k, h)  +  advected(3, level, k, h)  +   advected(4, level, k, h) +   advected(5, level, k, h) +   advected(6, level, k, h) +   advected(7, level, k, h) +   advected(8, level, k, h)) ;
            auto m1 = xt::eval(l1*(                                advected(1, level, k, h)                               -  advected(3, level, k, h)                               +   advected(5, level, k, h) -   advected(6, level, k, h) -   advected(7, level, k, h) +   advected(8, level, k, h)));
            auto m2 = xt::eval(l1*(                                                             advected(2, level, k, h)                               -   advected(4, level, k, h) +   advected(5, level, k, h) +   advected(6, level, k, h) -   advected(7, level, k, h) -   advected(8, level, k, h)));
            auto m3 = xt::eval(l2*(-4*advected(0, level, k, h) -   advected(1, level, k, h) -   advected(2, level, k, h)  -   advected(3, level, k, h) -   advected(4, level, k, h) + 2*advected(5, level, k, h) + 2*advected(6, level, k, h) + 2*advected(7, level, k, h) + 2*advected(8, level, k, h)));
            auto m4 = xt::eval(l3*(                            - 2*advected(1, level, k, h)                               + 2*advected(3, level, k, h)                              +   advected(5, level, k, h) -   advected(6, level, k, h)   - advected(7, level, k, h) +   advected(8, level, k, h)));
            auto m5 = xt::eval(l3*(                                                         - 2*advected(2, level, k, h)                               + 2*advected(4, level, k, h)   + advected(5, level, k, h) +   advected(6, level, k, h)   - advected(7, level, k, h) -   advected(8, level, k, h)));
            auto m6 = xt::eval(l4*( 4*advected(0, level, k, h) - 2*advected(1, level, k, h) - 2*advected(2, level, k, h)  - 2*advected(3, level, k, h) - 2*advected(4, level, k, h)   + advected(5, level, k, h) +   advected(6, level, k, h)   + advected(7, level, k, h) +   advected(8, level, k, h)));
            auto m7 = xt::eval(l2*(                                advected(1, level, k, h) -   advected(2, level, k, h)  +   advected(3, level, k, h) -   advected(4, level, k, h)                            ));
            auto m8 = xt::eval(l2*(                                                                                                                                                     advected(5, level, k, h) -   advected(6, level, k, h) +   advected(7, level, k, h) -   advected(8, level, k, h)));

            double space_step = 1.0 / (1 << max_level);
            double dummy = 3.0/(lambda*space_step);

            double cs2 = (lambda * lambda) / 3.0; // sound velocity squared
            double sigma_1 = dummy*zeta;
            double sigma_2 = dummy*mu;
            double s_1 = 1/(.5+sigma_1);
            double s_2 = 1/(.5+sigma_2);

            m3 = (1. - s_1) * m3 + s_1 * (-2*lambda*lambda*m0 + 3./m0*(m1*m1 + m2*m2));
            m4 = (1. - s_1) * m4 + s_1 * (-lambda*lambda*m1);
            m5 = (1. - s_1) * m5 + s_1 * (-lambda*lambda*m2);
            m6 = (1. - s_1) * m6 + s_1 * (lambda*lambda*lambda*lambda*m0 - 3.*lambda*lambda/m0*(m1*m1 + m2*m2));
            m7 = (1. - s_2) * m7 + s_2 * ((m1*m1-m2*m2)/m0);
            m8 = (1. - s_2) * m8 + s_2 * (m1*m2/m0);

            f(0, level, k, h) = (1./9)*m0                                  -  (1./9)*r2*m3                                +   (1./9)*r4*m6                         ;
            f(1, level, k, h) = (1./9)*m0   + (1./6)*r1*m1                 - (1./36)*r2*m3 - (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ;
            f(2, level, k, h) = (1./9)*m0                  +  (1./6)*r1*m2 - (1./36)*r2*m3                -  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ;
            f(3, level, k, h) = (1./9)*m0   - (1./6)*r1*m1                 - (1./36)*r2*m3 + (1./6)*r3*m4                 -  (1./18)*r4*m6 + .25*r2*m7             ;
            f(4, level, k, h) = (1./9)*m0                  -  (1./6)*r1*m2 - (1./36)*r2*m3                +  (1./6)*r3*m5 -  (1./18)*r4*m6 - .25*r2*m7             ;
            f(5, level, k, h) = (1./9)*m0   + (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ;
            f(6, level, k, h) = (1./9)*m0   - (1./6)*r1*m1 +  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 + (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ;
            f(7, level, k, h) = (1./9)*m0   - (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 -(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             + .25*r2*m8 ;
            f(8, level, k, h) = (1./9)*m0   + (1./6)*r1*m1 -  (1./6)*r1*m2 + (1./18)*r2*m3 +(1./12)*r3*m4 - (1./12)*r3*m5 +  (1./36)*r4*m6             - .25*r2*m8 ;
        });
    }
}

template<class Field>
void save_solution(const Field & f, const double t, const double c0, const std::size_t ite, const std::string ext = "")
{
    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::stringstream str;
    str << "D2Q9_mesh_jump_compte_rendus_"<<ext<<"_ite-" << ite;

    auto rho = samurai::make_field<double, 1>("rho", mesh);
    auto rho_ex = samurai::make_field<double, 1>("rho_ex", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4]
                               + f[cell][5] + f[cell][6] + f[cell][7] + f[cell][8];

        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        // rho_ex[cell] = exact_solution(x, y, t, c0);
    });
    samurai::save(str.str().data(), mesh, rho, rho_ex);
}

template<class Field>
double compute_error(const Field & f, const double t, const double c0)
{
    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    double diff = 0.;
    double normalization = 0.;

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        auto level = cell.level;

        double rho = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4]
                                + f[cell][5] + f[cell][6] + f[cell][7] + f[cell][8];

        const double rho_prime = rho - 1.;

        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        const double rho_prime_exact = exact_solution(x, y, t, c0, true);

        double dx = 1./(1<<level);

        diff += dx*dx*std::pow(rho_prime - rho_prime_exact, 2.);
        normalization += dx*dx*std::pow(rho_prime_exact, 2.);
    });
    return std::sqrt(diff / normalization);
}

template<class FieldRef, class FieldCoarse, class FieldJump, class Func>
std::array<double, 6> compute_errors(const FieldRef & fref, const FieldCoarse & fcoarse, const FieldJump & fjump,
                      const double t, const double c0, Func && update_bc_for_level)
{
    auto mesh_ref = fref.mesh();
    using mesh_id_t = typename decltype(mesh_ref)::mesh_id_t;
    auto mesh_coarse = fcoarse.mesh();
    auto mesh_jump = fjump.mesh();

    auto max_level = mesh_jump.max_level();
    auto min_level = mesh_jump.min_level();

    // We construct the exact solution once
    auto rho_ex = samurai::make_field<double, 1>("rho_ex", mesh_ref);
    samurai::for_each_cell(mesh_ref[mesh_id_t::cells], [&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];
        rho_ex[cell] = exact_solution(x, y, t, c0, true);
    });

    // We construct the conserved moment rho for each scheme
    auto rho_ref = samurai::make_field<double, 1>("rho_ref", mesh_ref);
    auto rho_coarse = samurai::make_field<double, 1>("rho_coarse", mesh_coarse);
    auto rho_jump = samurai::make_field<double, 1>("rho_jump", mesh_jump);

    samurai::for_each_cell(mesh_ref[mesh_id_t::cells], [&](auto &cell) {
        rho_ref[cell] = fref[cell][0] + fref[cell][1] + fref[cell][2] + fref[cell][3] + fref[cell][4]
                                      + fref[cell][5] + fref[cell][6] + fref[cell][7] + fref[cell][8] - 1.;
    });
    samurai::for_each_cell(mesh_coarse[mesh_id_t::cells], [&](auto &cell) {
        rho_coarse[cell] = fcoarse[cell][0] + fcoarse[cell][1] + fcoarse[cell][2] + fcoarse[cell][3] + fcoarse[cell][4]
                                            + fcoarse[cell][5] + fcoarse[cell][6] + fcoarse[cell][7] + fcoarse[cell][8] - 1.;
    });
    samurai::for_each_cell(mesh_jump[mesh_id_t::cells], [&](auto &cell) {
        rho_jump[cell] = fjump[cell][0] + fjump[cell][1] + fjump[cell][2] + fjump[cell][3] + fjump[cell][4]
                                        + fjump[cell][5] + fjump[cell][6] + fjump[cell][7] + fjump[cell][8] - 1.;
    });

    double l1_norm_solution = 0.; // Norm of the solution

    double l1_norm_error_ref = 0.; // Error of the reference scheme

    double l1_norm_error_coarse = 0.; // Error of the coarse scheme
    double l1_norm_diff_coarse = 0.; // Difference of the coarse with the reference scheme

    double l1_norm_error_jump = 0.; // Error of the jump scheme
    double l1_norm_diff_jump = 0.; // Difference of the jump scheme with the reference scheme
    double l1_norm_diff_jump_refl = 0.; // Difference of the jump scheme with the reference scheme only inside the central band

    auto exp = samurai::intersection(mesh_ref[mesh_id_t::cells][max_level], mesh_ref[mesh_id_t::cells][max_level]);
    exp([&](auto &interval, auto & index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        l1_norm_solution += xt::sum(xt::abs(rho_ex(max_level, k, h)))[0];
        l1_norm_error_ref += xt::sum(xt::abs(rho_ex(max_level, k, h) - rho_ref(max_level, k, h) ))[0];
    });

    // We reconstruct the densities of the schemes
    // not at the finest level on the finest level

    // We first update the ghosts
    samurai::update_ghost_mr(rho_coarse, std::forward<Func>(update_bc_for_level));
    samurai::update_ghost_mr(rho_jump, std::forward<Func>(update_bc_for_level));
    prepare_ghosts_one_field(rho_jump);

    auto rho_coarse_reconstructed = samurai::make_field<double, 1>("rho_coarse_reconstructed", mesh_ref);
    auto rho_jump_reconstructed = samurai::make_field<double, 1>("rho_jump_reconstructed", mesh_ref);
    auto level_jump = samurai::make_field<std::size_t, 1>("level_jump", mesh_ref); // levels

    // For the coarse mesh, everything is to be reconstructed
    auto to_predict_coarse = samurai::intersection(mesh_ref[mesh_id_t::cells][max_level], mesh_ref[mesh_id_t::cells][max_level]);

    using mesh_t = typename FieldRef::mesh_t;
    using interval_t = typename mesh_t::interval_t;

    to_predict_coarse([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        int parity_y = ((h & 1) == 1) ? -1 : 1;
        auto father_y = h >> 1;

        for (int k_cell = k.start; k_cell < k.end; ++k_cell)    {
            auto father_x = k_cell >> 1;
            interval_t father_int {father_x, father_x+1};
            int parity_x = ((k_cell & 1) == 1) ? -1 : 1;

                // // With the Haar wavelet
                // f(field_num, max_level, {k_cell, k_cell + 1}, h) = f(field_num, min_level, father_int, father_y) ;
                rho_coarse_reconstructed(max_level, {k_cell, k_cell + 1}, h) = rho_coarse(min_level, father_int, father_y)
                + 1./8*parity_x*(rho_coarse( min_level, father_int - 1, father_y) - rho_coarse( min_level, father_int + 1, father_y))
                + 1./8*parity_y*(rho_coarse( min_level, father_int, father_y - 1) - rho_coarse( min_level, father_int, father_y + 1))
                - 1./64*parity_x*parity_y*(rho_coarse(min_level, father_int+1, father_y + 1)+rho_coarse(min_level, father_int-1, father_y - 1)
                                          -rho_coarse(min_level, father_int+1, father_y - 1)-rho_coarse(min_level, father_int-1, father_y + 1));
        }
    });

    // For the jump mesh, we have to copy what is on the central band
    auto to_copy_jump = samurai::intersection(mesh_jump[mesh_id_t::cells][max_level], mesh_jump[mesh_id_t::cells][max_level]);
    to_copy_jump([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y
        rho_jump_reconstructed(max_level, k, h) = rho_jump(max_level, k, h);
        level_jump(max_level, k, h) = max_level;
    });
    // and to predict on the lateral ones
    auto to_predict_jump = samurai::intersection(mesh_jump[mesh_id_t::cells][min_level], mesh_jump[mesh_id_t::cells][min_level]).on(max_level);
    to_predict_jump([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        int parity_y = ((h & 1) == 1) ? -1 : 1;
        auto father_y = h >> 1;

        level_jump(max_level, k, h) = min_level;

        for (int k_cell = k.start; k_cell < k.end; ++k_cell)    {
            auto father_x = k_cell >> 1;
            interval_t father_int {father_x, father_x+1};
            int parity_x = ((k_cell & 1) == 1) ? -1 : 1;

            // // With the Haar wavelet
            // f(field_num, max_level, {k_cell, k_cell + 1}, h) = f(field_num, min_level, father_int, father_y) ;
            rho_jump_reconstructed(max_level, {k_cell, k_cell + 1}, h) = rho_jump(min_level, father_int, father_y)
            + 1./8*parity_x*(rho_jump( min_level, father_int - 1, father_y) - rho_jump( min_level, father_int + 1, father_y))
            + 1./8*parity_y*(rho_jump( min_level, father_int, father_y - 1) - rho_jump( min_level, father_int, father_y + 1))
            - 1./64*parity_x*parity_y*(rho_jump(min_level, father_int+1, father_y + 1)+rho_jump(min_level, father_int-1, father_y - 1)
                                      -rho_jump(min_level, father_int+1, father_y - 1)-rho_jump(min_level, father_int-1, father_y + 1));
        }
    });

    samurai::save(std::string("D2Q9_mesh_jump_compte_rendus_reflected_wave"), mesh_ref, rho_ref, rho_jump_reconstructed, level_jump);

    // samurai::save(std::string("D2Q9_mesh_jump_compte_rendus_DBG"), mesh_ref, rho_jump_reconstructed);

    // We can compute the remaining errors
    to_predict_coarse([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        l1_norm_error_coarse += xt::sum(xt::abs(rho_ex(max_level, k, h) - rho_coarse_reconstructed(max_level, k, h)))[0];
        l1_norm_diff_coarse  += xt::sum(xt::abs(rho_ref(max_level, k, h) - rho_coarse_reconstructed(max_level, k, h)))[0];

        l1_norm_error_jump += xt::sum(xt::abs(rho_ex(max_level, k, h)  - rho_jump_reconstructed(max_level, k, h)))[0];
        l1_norm_diff_jump  += xt::sum(xt::abs(rho_ref(max_level, k, h) - rho_jump_reconstructed(max_level, k, h)))[0];
    });
    // Just the reflected error is different
    to_copy_jump([&](auto& interval, auto& index) {
        auto k = interval; // Logical index in x
        auto h = index[0];    // Logical index in y

        l1_norm_diff_jump_refl  += xt::sum(xt::abs(rho_ref(max_level, k, h) - rho_jump_reconstructed(max_level, k, h)))[0];
    });

    return {l1_norm_error_ref / l1_norm_solution,
            l1_norm_error_coarse / l1_norm_solution,
            l1_norm_diff_coarse / l1_norm_solution,
            l1_norm_error_jump / l1_norm_solution,
            l1_norm_diff_jump / l1_norm_solution,
            l1_norm_diff_jump_refl / l1_norm_solution};
}


template<class FieldRef, class FieldJump>
void save_solution_cut(const FieldRef & fref, const FieldJump & fjump, const double t, const double c0, const std::string ext = "")
{
    auto mesh_ref = fref.mesh();
    auto mesh_jump = fjump.mesh();

    using mesh_id_t = typename decltype(mesh_ref)::mesh_id_t;
    auto max_level = mesh_ref.max_level();
    auto min_level = mesh_ref.min_level();

    // We start by the exact and the reference solution
    std::ofstream x_reference;
    std::ofstream y_reference;
    std::ofstream y_exact;
    x_reference.open("./D2Q9_jump_compte_rendus/x_reference_"+ext+".dat");
    y_reference.open("./D2Q9_jump_compte_rendus/y_reference_"+ext+".dat");
    y_exact.open("./D2Q9_jump_compte_rendus/y_exact_"+ext+".dat");

    const int h = 0;
    const double dx = 1./(1<<max_level);
    const double y_point = h*dx + .5*dx;

    // We construct the conserved moment
    auto rho_ref = samurai::make_field<double, 1>("rho_ref", mesh_ref);

    samurai::for_each_cell(mesh_ref[mesh_id_t::cells], [&](auto &cell) {
        rho_ref[cell] = fref[cell][0] + fref[cell][1] + fref[cell][2] + fref[cell][3] + fref[cell][4]
                                      + fref[cell][5] + fref[cell][6] + fref[cell][7] + fref[cell][8] - 1.;
    });

    for (int k = -static_cast<int>(1<<max_level); k < static_cast<int>(1<<max_level); ++k)  {
        double x_point = k*dx + .5*dx;
        x_reference<<x_point<<std::endl;
        y_reference<<rho_ref(max_level, {k, k+1}, h)[0]<<std::endl;
        y_exact<<exact_solution(x_point, y_point, t, c0, true)<<std::endl;
    }
    x_reference.close();
    y_reference.close();
    y_exact.close();

    // Now we go to the solution with the jump
    std::ofstream x_jump;
    std::ofstream y_jump;
    x_jump.open("./D2Q9_jump_compte_rendus/x_jump_"+ext+".dat");
    y_jump.open("./D2Q9_jump_compte_rendus/y_jump_"+ext+".dat");

    auto rho_jump = samurai::make_field<double, 1>("rho_jump", mesh_jump);

    samurai::for_each_cell(mesh_jump[mesh_id_t::cells], [&](auto &cell) {
        rho_jump[cell] = fjump[cell][0] + fjump[cell][1] + fjump[cell][2] + fjump[cell][3] + fjump[cell][4]
                                        + fjump[cell][5] + fjump[cell][6] + fjump[cell][7] + fjump[cell][8] - 1.;
    });

    for (std::size_t level = min_level; level <= max_level; ++level)    {
        auto leaves = samurai::intersection(mesh_jump[mesh_id_t::cells][level], mesh_jump[mesh_id_t::cells][level]);
        const double dx_loc = 1./(1<<level);

        leaves([&](auto& interval, auto& index) {
            auto k = interval; // Logical index in x
            auto h_loc = index[0];    // Logical index in y

            if (h_loc == h) {
                for (int k_loc = k.start; k_loc < k.end; ++k_loc)   {
                    double x_point = k_loc*dx_loc + .5*dx_loc;
                    x_jump<<x_point<<std::endl;
                    y_jump<<rho_jump(level, {k_loc, k_loc + 1}, h_loc)[0]<<std::endl;
                }
            }
        });
    }
    x_jump.close();
    y_jump.close();
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim, 2>;
    using mesh_t = samurai::MRMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type = typename mesh_t::cl_type;

    using coord_index_t = typename mesh_t::interval_t::coord_index_t;

    const double mu = 1.5e-5; // Bulk viscosity
    const double zeta = 1. * mu; // Shear viscosity

    const size_t max_level = 8;
    const size_t min_level = max_level - 1;

    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;
    using coord_index_t = typename samurai::MRMesh<Config>::coord_index_t;
    auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);

    const double lambda = 1.;

    cl_type cell_list_jump;
    cl_type cell_list_ref;
    cl_type cell_list_coarse;

    // for (int y = -(1<<max_level); y < (1<<max_level); ++y)  {
        // cell_list[max_level][{y}].add_interval({-256, 256});
    // }
    // for (int y = -(1<<min_level); y < (1<<min_level); ++y)  {
        // cell_list[min_level][{y}].add_interval({-128, 128});
    // }
    // We add the central band
    for (int y = -(1<<max_level); y < (1<<max_level); ++y)  {
        cell_list_jump[max_level][{y}].add_interval({-static_cast<int>(1<<max_level)*13/32, static_cast<int>(1<<max_level)*13/32});
        cell_list_ref[max_level][{y}].add_interval({-static_cast<int>(1<<max_level), static_cast<int>(1<<max_level)});
    }
    // We add the two lateral bands
    for (int y = -(1<<min_level); y < (1<<min_level); ++y)  {
        cell_list_jump[min_level][{y}].add_interval({-static_cast<int>(1<<min_level), -static_cast<int>(1<<min_level)*13/32});
        cell_list_jump[min_level][{y}].add_interval({static_cast<int>(1<<min_level)*13/32, static_cast<int>(1<<min_level)});
        cell_list_coarse[min_level][{y}].add_interval({-static_cast<int>(1<<min_level), static_cast<int>(1<<min_level)});
    }

    mesh_t mesh_jump(cell_list_jump, min_level, max_level);
    mesh_t mesh_ref(cell_list_ref, min_level, max_level);
    mesh_t mesh_coarse(cell_list_coarse, min_level, max_level);

    auto f_fields_jump = samurai::make_field<double, 9>("f", mesh_jump);
    auto f_fields_ref = samurai::make_field<double, 9>("f", mesh_ref);
    auto f_fields_coarse = samurai::make_field<double, 9>("f", mesh_coarse);

    init_fields(f_fields_jump, lambda);
    init_fields(f_fields_ref, lambda);
    init_fields(f_fields_coarse, lambda);

    save_solution(f_fields_jump, 0., lambda/(std::sqrt(3.)), 0, "jump");
    save_solution(f_fields_ref, 0., lambda/(std::sqrt(3.)), 0, "ref");
    save_solution(f_fields_coarse, 0., lambda/(std::sqrt(3.)), 0, "coarse");

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        update_bc_D2Q4_3_Euler_constant_extension(field, level);
    };
    samurai::update_ghost_mr(f_fields_jump, update_bc_for_level);

    double t = 0.;
    double dx = 1./(1<<max_level);
    double dt = dx / lambda;

    for (std::size_t ite = 0; ite < (1<<max_level); ++ite) {
        std::cout<<"Iteration = "<<ite<<std::endl;
        // save_solution(f_fields_jump, t, lambda/(std::sqrt(3.)), ite);
        one_time_step(f_fields_jump, update_bc_for_level, pred_coeff, lambda, mu, zeta, ite);
        one_time_step(f_fields_ref, update_bc_for_level, pred_coeff, lambda, mu, zeta, ite);
        one_time_step(f_fields_coarse, update_bc_for_level, pred_coeff, lambda, mu, zeta, ite);

        t += dt;

        if (ite == static_cast<std::size_t>((1<<max_level)/10.))
            save_solution_cut(f_fields_ref, f_fields_jump, t, lambda/(std::sqrt(3.)), "begin");
    }

    std::cout<<std::endl;

    save_solution_cut(f_fields_ref, f_fields_jump, t, lambda/(std::sqrt(3.)), "final");

    auto errors = compute_errors(f_fields_ref, f_fields_coarse, f_fields_jump, t, lambda/(std::sqrt(3.)), update_bc_for_level);

    std::cout<<errors[0]<<"\t"
             <<errors[1]<<"\t"
             <<errors[2]<<"\t"
             <<errors[3]<<"\t"
             <<errors[4]<<"\t"
             <<errors[5]<<"\t"<<std::endl;


    // samurai::save(std::string("D2Q9_mesh_jump_compte_rendus"), {true, true}, mesh, f_fields);

    return 0;
}
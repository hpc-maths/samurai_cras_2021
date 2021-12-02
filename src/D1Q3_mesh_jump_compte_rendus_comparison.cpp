// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <math.h>
#include <vector>
#include <fstream>

#include <cxxopts.hpp>

#include <xtensor/xio.hpp>

#include <samurai/mr/adapt.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/hdf5.hpp>

#include "prediction_map_1d.hpp"
#include "boundary_conditions.hpp"

#include "utils_lbm_mr_1d.hpp"

#include <chrono>

template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(2);

        data[k][0] = prediction(k, i*size - 1) - prediction(k, (i+1)*size - 1);
        data[k][1] = prediction(k, (i+1)*size) - prediction(k, i*size);
    }

    return data;
}

template<class Field, class interval_t>
xt::xtensor<double, 1> prediction_naive(const Field & f, std::size_t field_num, interval_t k, std::size_t min_level, std::size_t dl)
{
    if (dl == 0)
        return f(field_num, min_level, k);
    else
    {
        // Check for odd indice
        int coeff = (k.start & 1) ? -1 : 1;
        auto father = k/2;

        return xt::eval(prediction_naive(f, field_num, father, min_level, dl-1) + 1./8*coeff*(prediction_naive(f, field_num, father-1, min_level, dl-1)
                                                                                             -prediction_naive(f, field_num, father+1, min_level, dl-1)));
    }
}

double u0 (const double x)
{
    return std::exp(-100*std::pow(x - 3./2, 2.));
}

double sol_u(const double t, const double x, const double c)
{
    return .5*u0(x - c*t) + .5*u0(x + c*t);
}

template<class Field>
void init_fields(Field & f, const double lambda, const double c)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    auto mesh = f.mesh();
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();
        auto x = center[0];

        double u = u0(x);
        double v = 0.;
        double w = .5*c*c*u;

        f[cell][0] = u                    - 2./(lambda*lambda) * w;
        f[cell][1] =    1./(2*lambda) * v + 1./(lambda*lambda) * w;
        f[cell][2] =   -1./(2*lambda) * v + 1./(lambda*lambda) * w;
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
    const auto max_level = mesh.max_level();
    const auto dl = max_level - min_level;

    // We first update the coarse cells beneath the refined ones

    auto to_project = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][min_level],
                                            mesh[mesh_id_t::cells][max_level]).on(min_level);
    to_project([&](auto& interval, auto) {
        auto k = interval; // Logical index in x

        for (int k_cell = k.start; k_cell < k.end; ++k_cell)    {
            const int k_fine_start =  k_cell      * (1<<dl);
            const int k_fine_end   = (k_cell + 1) * (1<<dl);

            for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {
                f(field_num, min_level, {k_cell, k_cell + 1}) = xt::mean(f(field_num, max_level, {k_fine_start, k_fine_end}));
            }
        }
    });

    // We eventually update the cells at the finest level above the coarse ones

    auto to_predict = samurai::intersection(samurai::difference(mesh[mesh_id_t::cells_and_ghosts][max_level],
                                                                mesh[mesh_id_t::cells][max_level]),
                                            mesh.domain());

    std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;

    to_predict([&](auto& interval, auto) {
        auto k = interval; // Logical index in x

        // k.step = (1 << dl);

        for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {
            // f(field_num, max_level, k) = prediction(f, min_level, dl, k, field_num, memoization_map);
            for (int idx = k.start; idx < k.end; ++idx) {
                interval_t target_int{idx, idx+1};
                f(field_num, max_level, target_int) = prediction_naive(f, field_num, target_int, min_level, dl);
            }
        }
    });
}

template<class Field, class Func>
void one_time_step_Rohde(Field & f, Func && update_bc_for_level, const double lambda, const double p, const double c)
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

    auto advected_f = samurai::make_field<double, nvel>("advected_f", mesh);
    advected_f.fill(0.);

    // 1 - Collision everywhere
    for (std::size_t level = 0; level <= max_level; ++level)    {

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);
        leaves([&](auto &interval, auto) {
            auto k = interval;

            auto u = xt::eval(f(0, level, k)  + f(1, level, k) + f(2, level, k) );
            auto v = xt::eval(lambda*(          f(1, level, k) - f(2, level, k)));
            auto w = xt::eval(.5*lambda*lambda*(f(1, level, k) + f(2, level, k)));

            w = (1. - p)*w + p*(.5*c*c*u);

            f(0, level, k) = xt::eval(u                    - 2./(lambda*lambda) * w);
            f(1, level, k) = xt::eval(   1./(2*lambda) * v + 1./(lambda*lambda) * w);
            f(2, level, k) = xt::eval(  -1./(2*lambda) * v + 1./(lambda*lambda) * w);
        });
    }

    // 2 - Homogeneous redistribution from coarse to fine

    double foo = 0.;

    xt::xarray<double>::shape_type shape = {nvel, static_cast<std::size_t>(1<<dl + 2)};
    xt::xarray<double> f_fine_over_coarse(shape);

    const xt::xtensor_fixed<int, xt::xshape<1>> xp{1};
    auto fine_over_coarse = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][max_level], xp),
                                                  mesh[mesh_id_t::cells][min_level]);

    fine_over_coarse([&](auto& interval, auto) {
        auto k = interval; // Logical index in x
        interval_t father {k.start/(1<<dl), k.start/(1<<dl) + 1};

        for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {

            f_fine_over_coarse(field_num, 0) = f(field_num, max_level, k - 1)[0];
            for (std::size_t shift = 0; shift < (1<<dl); ++shift)   {
                f_fine_over_coarse(field_num, 1+shift) = f(field_num, min_level, father)[0];
            }
            f_fine_over_coarse(field_num, 1 + (1<<dl)) = f(field_num, min_level, father+1)[0];
            foo =  f(field_num, min_level, father+1)[0];
        }
    });

    // // 3 - Stream on the fine cells and on the fine cells on top of coarse cells and on the coarse cells
    auto last_fine = samurai::difference(mesh[mesh_id_t::cells][max_level],
                                         samurai::translate(mesh[mesh_id_t::cells][max_level], -xp));

    auto fine_except_last = samurai::difference(mesh[mesh_id_t::cells][max_level], last_fine);

    fine_except_last([&](auto& interval, auto) {
        auto k = interval; // Logical index in x

        advected_f(0, max_level, k) = f(0, max_level, k    );
        advected_f(1, max_level, k) = f(1, max_level, k - 1);
        advected_f(2, max_level, k) = f(2, max_level, k + 1);
    });

    last_fine([&](auto& interval, auto) {
        auto k = interval; // Logical index in x
        advected_f(0, max_level, k) = f(0, max_level, k    );
        advected_f(1, max_level, k) = f(1, max_level, k - 1);
        advected_f(2, max_level, k) = f_fine_over_coarse(2, 1);
    });

    xt::xarray<double> f_fine_over_coarse_advected(shape);
    for (std::size_t shift = 0; shift < (1<<dl); ++shift)   {
        f_fine_over_coarse_advected(0, 1 + shift) = f_fine_over_coarse(0, 1 + shift);
        f_fine_over_coarse_advected(1, 1 + shift) = f_fine_over_coarse(1, 1 + shift - 1);
        f_fine_over_coarse_advected(2, 1 + shift) = f_fine_over_coarse(2, 1 + shift + 1);
    }

    // We prepare the ghost cells for the coarse mesh
    auto to_project = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][min_level],
                                            mesh[mesh_id_t::cells][max_level]).on(min_level);
    to_project([&](auto& interval, auto) {
        auto k = interval; // Logical index in x

        for (int k_cell = k.start; k_cell < k.end; ++k_cell)    {
            const int k_fine_start =  k_cell      * (1<<dl);
            const int k_fine_end   = (k_cell + 1) * (1<<dl);

            for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {
                f(field_num, min_level, {k_cell, k_cell + 1}) = xt::mean(f(field_num, max_level, {k_fine_start, k_fine_end}));
            }
        }
    });

    auto coarse_cells = samurai::intersection(mesh[mesh_id_t::cells][min_level], mesh[mesh_id_t::cells][min_level]);
    coarse_cells([&](auto& interval, auto) {
        auto k = interval; // Logical index in x

        advected_f(0, min_level, k) = f(0, min_level, k    );
        advected_f(1, min_level, k) = f(1, min_level, k - 1);
        advected_f(2, min_level, k) = f(2, min_level, k + 1);
    });

    std::swap(f.array(), advected_f.array());
    fine_over_coarse([&](auto& interval, auto) {
        auto k = interval; // Logical index in x
        interval_t father {k.start/(1<<dl), k.start/(1<<dl) + 1};

        for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {

            f_fine_over_coarse_advected(field_num, 0) = f(field_num, max_level, k - 1)[0];
            // f_fine_over_coarse_advected(field_num, 1 + (1<<dl)) = f(field_num, min_level, father+1)[0];
            f_fine_over_coarse_advected(field_num, 1 + (1<<dl)) = foo;
        }
    });
    f_fine_over_coarse = f_fine_over_coarse_advected;

    // 4
    auto leaves_fine = samurai::intersection(mesh[mesh_id_t::cells][max_level],
                                             mesh[mesh_id_t::cells][max_level]);

    for (std::size_t rpt = 0; rpt < ((1<<dl)-1); ++rpt) {
        leaves_fine([&](auto &interval, auto) {
            auto k = interval;

            auto u = xt::eval(f(0, max_level, k)  + f(1, max_level, k) + f(2, max_level, k) );
            auto v = xt::eval(lambda*(              f(1, max_level, k) - f(2, max_level, k)));
            auto w = xt::eval(.5*lambda*lambda*(    f(1, max_level, k) + f(2, max_level, k)));

            w = (1. - p)*w + p*(.5*c*c*u);

            f(0, max_level, k) = xt::eval(u                    - 2./(lambda*lambda) * w);
            f(1, max_level, k) = xt::eval(   1./(2*lambda) * v + 1./(lambda*lambda) * w);
            f(2, max_level, k) = xt::eval(  -1./(2*lambda) * v + 1./(lambda*lambda) * w);
        });

        fine_except_last([&](auto& interval, auto) {
            auto k = interval; // Logical index in x

            advected_f(0, max_level, k) = f(0, max_level, k    );
            advected_f(1, max_level, k) = f(1, max_level, k - 1);
            advected_f(2, max_level, k) = f(2, max_level, k + 1);
        });
        last_fine([&](auto& interval, auto) {
            auto k = interval; // Logical index in x
            advected_f(0, max_level, k) = f(0, max_level, k    );
            advected_f(1, max_level, k) = f(1, max_level, k - 1);
            advected_f(2, max_level, k) = f_fine_over_coarse(2, 1);
        });

        // xt::xarray<double> f_fine_over_coarse_advected(shape);
        for (std::size_t shift = 0; shift < (1<<dl); ++shift)   {
            f_fine_over_coarse_advected(0, 1 + shift) = f_fine_over_coarse(0, 1 + shift);
            f_fine_over_coarse_advected(1, 1 + shift) = f_fine_over_coarse(1, 1 + shift - 1);
            f_fine_over_coarse_advected(2, 1 + shift) = f_fine_over_coarse(2, 1 + shift + 1);
        }

        coarse_cells([&](auto& interval, auto) {
            auto k = interval; // Logical index in x

            advected_f(0, min_level, k) = f(0, min_level, k);
            advected_f(1, min_level, k) = f(1, min_level, k);
            advected_f(2, min_level, k) = f(2, min_level, k);
        });

        std::swap(f.array(), advected_f.array());
        fine_over_coarse([&](auto& interval, auto) {
            auto k = interval; // Logical index in x
            interval_t father {k.start/(1<<dl), k.start/(1<<dl) + 1};

            for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {

                f_fine_over_coarse_advected(field_num, 0) = f(field_num, max_level, k - 1)[0];
                // f_fine_over_coarse_advected(field_num, 1 + (1<<dl)) = f(field_num, min_level, father+1)[0];
                f_fine_over_coarse_advected(field_num, 1 + (1<<dl)) = foo;

            }
        });
        f_fine_over_coarse = f_fine_over_coarse_advected;
    }

    // 5
    auto coarse_with_fine_ontop = samurai::intersection(mesh[mesh_id_t::cells][min_level], mesh[mesh_id_t::cells_and_ghosts][max_level]).on(min_level);
    coarse_with_fine_ontop([&](auto& interval, auto) {
        auto k = interval; // Logical index in x

        f(1, min_level, k) = xt::mean(xt::view(f_fine_over_coarse, 1, xt::range(1, (1<<dl)+1)));

        // for (std::size_t field_num = 0; field_num < nvel; ++field_num)  {
            // f(field_num, min_level, k) = xt::mean(xt::view(f_fine_over_coarse, field_num, xt::range(1, (1<<dl)+1)));
        // }
    });
}

template<class Field, class Func, class PredCoeff>
void one_time_step(Field & f, Func && update_bc_for_level, const PredCoeff & pred_coeff, const double lambda, const double p, const double c, const bool is_jump = false, const bool lax_wendroff = false)
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

    auto advected_f = samurai::make_field<double, nvel>("advected_f", mesh);
    advected_f.fill(0.);

    // samurai::mr_projection(f);
    // for (std::size_t level = min_level - 1; level <= max_level; ++level)
    // {
    //     update_bc_for_level(f, level);
    // }
    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    // It is not capable of doing so.
    // samurai::mr_prediction(f, update_bc_for_level);

    // If the mesh has a jump, thus is not uniform
    // we have to update the ghosts
    if (is_jump)    {
        prepare_ghosts(f);
    }

    // Stream
    for (std::size_t level = 0; level <= max_level; ++level)    {

        const std::size_t dl = max_level - level;
        const double coeff = 1. / (1 << dl);

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        leaves([&](auto& interval, auto) {
            auto k = interval; // Logical index in x

            auto f1 = xt::eval(f(1, level, k));
            auto f2 = xt::eval(f(2, level, k));

            if (!lax_wendroff)  {
                for(auto &c: pred_coeff[dl][0].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;
                    f1 += coeff * weight * f(1, level, k + stencil);
                }
                for(auto &c: pred_coeff[dl][1].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;
                    f2 += coeff * weight * f(2, level, k + stencil);
                }
            }
            else
            {
                f1 += -coeff*coeff*f(1, level, k) - .5*coeff*(1.-coeff)*f(1, level, k+1)+.5*coeff*(1+coeff)*f(1, level, k-1);
                f2 += -coeff*coeff*f(2, level, k) - .5*coeff*(1.-coeff)*f(2, level, k-1)+.5*coeff*(1+coeff)*f(2, level, k+1);
            }

            advected_f(0, level, k) = f(0, level, k);
            advected_f(1, level, k) = f1;
            advected_f(2, level, k) = f2;
        });

    }

    // Collision
    for (std::size_t level = 0; level <= max_level; ++level)    {

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);
        leaves([&](auto &interval, auto) {
            auto k = interval;

            auto u = xt::eval(advected_f(0, level, k) + advected_f(1, level, k) + advected_f(2, level, k) );
            auto v = xt::eval(lambda*(                  advected_f(1, level, k) - advected_f(2, level, k)));
            auto w = xt::eval(.5*lambda*lambda*(        advected_f(1, level, k) + advected_f(2, level, k)));

            w = (1. - p)*w + p*(.5*c*c*u);

            f(0, level, k) = xt::eval(u                    - 2./(lambda*lambda) * w);
            f(1, level, k) = xt::eval(   1./(2*lambda) * v + 1./(lambda*lambda) * w);
            f(2, level, k) = xt::eval(  -1./(2*lambda) * v + 1./(lambda*lambda) * w);
        });
    }
}

template<class Field>
void save_solution(const Field & f, const double t, const double c, const std::size_t ite, const std::string ext = "")
{
    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;


    std::stringstream str;
    str << "LBM_D1Q3_mesh_jump_compte_rendus_"<<ext<<"_ite-" << ite;

    auto u = samurai::make_field<double, 1>("u", mesh);
    auto u_ex = samurai::make_field<double, 1>("u_ex", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];

        u[cell] = f[cell][0] + f[cell][1] + f[cell][2];
        u_ex[cell] = sol_u(t, x, c);
    });

    samurai::save(str.str().data(), mesh, u, u_ex);
}


int main(int argc, char *argv[])
{
    cxxopts::Options options("D1Q3 with mesh jump for the Compte Rendus to plot",
                             "...");

    options.add_options()
                        ("level_diff", "level difference", cxxopts::value<std::size_t>()->default_value("3"))
                        ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("9"))
                        ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                        ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {
            constexpr size_t dim = 1;
            using Config = samurai::MRConfig<dim, 2>;
            using mesh_t = samurai::MRMesh<Config>;
            using mesh_id_t = typename mesh_t::mesh_id_t;
            using cl_type = typename mesh_t::cl_type;

            using coord_index_t = typename mesh_t::interval_t::coord_index_t;

            const std::size_t level_diff = result["level_diff"].as<std::size_t>();
            const std::size_t max_level = result["max_level"].as<std::size_t>();
            const auto min_level = max_level - level_diff;

            const double lambda = 1.;
            const double p = 1.7;
            const double T = 1.5625;
            const double c = 0.5;

            // Mesh for the scheme with jump
            cl_type cell_list_jump;
            cell_list_jump[max_level][{}].add_interval({0, 2*(1<<max_level)});
            cell_list_jump[min_level][{}].add_interval({2*(1<<min_level), 3*(1<<min_level)});
            mesh_t mesh_jump(cell_list_jump, min_level, max_level);

            auto f_jump = samurai::make_field<double, 3>("f", mesh_jump);
            init_fields(f_jump, lambda, c);

            auto f_jump_lw = samurai::make_field<double, 3>("f", mesh_jump);
            init_fields(f_jump_lw, lambda, c);

            auto f_jump_Rohde = samurai::make_field<double, 3>("f", mesh_jump);
            init_fields(f_jump_Rohde, lambda, c);

            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);
            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };

            double dx = 1. / (1<<max_level);
            double dt = dx / lambda;
            std::size_t N_ite = T / dt;
            double t = 0.;

            for (std::size_t it = 0; it < N_ite; ++it) {
                std::cout<<"Iteration = "<<it<<std::endl;

                // save_solution(f_jump, t   , c, it);
                // save_solution(f_jump_lw, t, c, it, std::string("LW"));
                save_solution(f_jump_Rohde, t, c, it, std::string("Rohde"));

                one_time_step(f_jump   , update_bc_for_level, pred_coeff, lambda, p, c, true, false);
                one_time_step(f_jump_lw, update_bc_for_level, pred_coeff, lambda, p, c, true, true);

                // Since the scheme by Rohde has a local time step, we had to call the
                // step function each 2^(level_difference) iterations
                if (it % (1<<level_diff) == 0)
                    one_time_step_Rohde(f_jump_Rohde, update_bc_for_level, lambda, p, c);

                t += dt;
            }
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}

// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <Eigen/Eigen>
#include "solver_engine.h"
#include "fundamental_estimator.h"
#include "/usr/local/include/eigen3/unsupported/Eigen/Polynomials"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class StitchingGravityRadialCalib : public SolverEngine
			{
			public:
				// StitchingGravityRadialCalib() : gravity_source(Eigen::Matrix3d::Identity()),
				// 								gravity_destination(Eigen::Matrix3d::Identity())
				// {
				// }

				~StitchingGravityRadialCalib()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 1;
				}

				static constexpr size_t maximumSolutions()
				{
					return 6;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr bool needsGravity()
				{
					return true;
				}

				static constexpr char *getName()
				{
					return "Gravity Equal Radial (2PC)";
				}

				void setGravity(const Eigen::Matrix3d &gravity_source_,
								const Eigen::Matrix3d &gravity_destination_)
				{
					gravity_source = gravity_source_;
					gravity_destination = gravity_destination_;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sample_number_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

				void solve_quadratic(double a, double b, double c, std::complex<double> roots[2]) const;

				Eigen::Matrix3d K1, K2;

			protected:
				Eigen::Matrix3d gravity_source;
				Eigen::Matrix3d gravity_destination;
			};

			void StitchingGravityRadialCalib::solve_quadratic(double a, double b, double c, std::complex<double> roots[2]) const
			{
				std::complex<double> b2m4ac = b * b - 4 * a * c;
				std::complex<double> sq = std::sqrt(b2m4ac);

				// Choose sign to avoid cancellations
				roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
				roots[1] = c / (a * roots[0]);
			}

			OLGA_INLINE bool StitchingGravityRadialCalib::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;

				if (sample_number_ < sampleSize())
					return false;

				// Building the coefficient matrices
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t cols = data_.cols;
				constexpr size_t pointNumber = sampleSize(); // 3

				const double
					&u1 = data_.at<double>(sample_[0], 0),
					&v1 = data_.at<double>(sample_[0], 1),
					&u2 = data_.at<double>(sample_[0], 2),
					&v2 = data_.at<double>(sample_[0], 3),

					if (gravity_source(0, 0) != 1)
				{
					double r1 = u1 * u1 + v1 * v1,
						   r2 = u2 * u2 + v2 * v2;

					double a1 = gravity_source(0, 1) / gravity_source(0, 0);
					double a2 = gravity_source(1, 0) / gravity_source(0, 0);
					double a3 = gravity_source(1, 1) / gravity_source(0, 0);
					double a4 = gravity_source(1, 2) / gravity_source(0, 0);
					double a5 = gravity_source(2, 0) / gravity_source(0, 0);
					double a6 = gravity_source(2, 1) / gravity_source(0, 0);
					double a7 = gravity_source(2, 2) / gravity_source(0, 0);

					double b1 = gravity_destination(0, 1) / gravity_destination(0, 0);
					double b2 = gravity_destination(1, 0) / gravity_destination(0, 0);
					double b3 = gravity_destination(1, 1) / gravity_destination(0, 0);
					double b4 = gravity_destination(1, 2) / gravity_destination(0, 0);
					double b5 = gravity_destination(2, 0) / gravity_destination(0, 0);
					double b6 = gravity_destination(2, 1) / gravity_destination(0, 0);
					double b7 = gravity_destination(2, 2) / gravity_destination(0, 0);

					double d[20] = {u1, v1, u2, v2, r1, r2, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7};

					VectorXd coeffs(15);
					coeffs[0] = d[4] * d[5] * d[9] * d[14] - d[4] * d[5] * d[12] * d[17];
					coeffs[1] = d[0] * d[5] * d[7] * d[14] + d[1] * d[5] * d[8] * d[14] - d[2] * d[4] * d[9] * d[16] - d[0] * d[5] * d[10] * d[17] - d[1] * d[5] * d[11] * d[17] + d[2] * d[4] * d[12] * d[19] - d[1] * d[5] * d[6] + d[4] * d[9] * d[14] + d[5] * d[9] * d[14] - d[4] * d[12] * d[17] - d[5] * d[12] * d[17] - d[0] * d[5];
					coeffs[2] = 2 * d[4] * d[5] * d[12];
					coeffs[3] = -d[0] * d[2] * d[7] * d[16] - d[1] * d[2] * d[8] * d[16] + d[0] * d[2] * d[10] * d[19] + d[1] * d[2] * d[11] * d[19] + d[0] * d[7] * d[14] + d[1] * d[8] * d[14] - d[2] * d[9] * d[16] - d[0] * d[10] * d[17] - d[1] * d[11] * d[17] + d[2] * d[12] * d[19] - d[1] * d[6] + d[9] * d[14] - d[12] * d[17] - d[0];
					coeffs[4] = -2 * d[1] * d[5] * d[6] * d[17] + 2 * d[0] * d[5] * d[10] + 2 * d[1] * d[5] * d[11] - 2 * d[0] * d[5] * d[17] + 2 * d[4] * d[12] + 2 * d[5] * d[12];
					coeffs[5] = d[4] * d[5] * d[9] * d[14] + d[4] * d[5] * d[12] * d[17];
					coeffs[6] = 2 * d[1] * d[2] * d[6] * d[19] - 2 * d[1] * d[6] * d[17] + 2 * d[0] * d[2] * d[19] + 2 * d[0] * d[10] + 2 * d[1] * d[11] - 2 * d[0] * d[17] + 2 * d[12];
					coeffs[7] = d[0] * d[5] * d[7] * d[14] + d[1] * d[5] * d[8] * d[14] - d[2] * d[4] * d[9] * d[16] + d[0] * d[5] * d[10] * d[17] + d[1] * d[5] * d[11] * d[17] - d[2] * d[4] * d[12] * d[19] + d[1] * d[5] * d[6] + d[4] * d[9] * d[14] + d[5] * d[9] * d[14] + d[4] * d[12] * d[17] + d[5] * d[12] * d[17] + d[0] * d[5];
					coeffs[8] = -d[0] * d[2] * d[7] * d[16] - d[1] * d[2] * d[8] * d[16] - d[0] * d[2] * d[10] * d[19] - d[1] * d[2] * d[11] * d[19] + d[0] * d[7] * d[14] + d[1] * d[8] * d[14] - d[2] * d[9] * d[16] + d[0] * d[10] * d[17] + d[1] * d[11] * d[17] - d[2] * d[12] * d[19] + d[1] * d[6] + d[9] * d[14] + d[12] * d[17] + d[0];
					coeffs[9] = -d[3] * d[4] * d[9] * d[14] + d[2] * d[4] * d[9] * d[15] + d[3] * d[4] * d[12] * d[17] - d[2] * d[4] * d[12] * d[18];
					coeffs[10] = -d[1] * d[2] * d[6] * d[13] - d[0] * d[3] * d[7] * d[14] - d[1] * d[3] * d[8] * d[14] + d[0] * d[2] * d[7] * d[15] + d[1] * d[2] * d[8] * d[15] + d[0] * d[3] * d[10] * d[17] + d[1] * d[3] * d[11] * d[17] - d[0] * d[2] * d[10] * d[18] - d[1] * d[2] * d[11] * d[18] + d[1] * d[3] * d[6] - d[0] * d[2] * d[13] - d[3] * d[9] * d[14] + d[2] * d[9] * d[15] + d[3] * d[12] * d[17] - d[2] * d[12] * d[18] + d[0] * d[3];
					coeffs[11] = 2 * d[2] * d[4] * d[12] * d[13] - 2 * d[3] * d[4] * d[12];
					coeffs[12] = 2 * d[0] * d[2] * d[10] * d[13] + 2 * d[1] * d[2] * d[11] * d[13] + 2 * d[1] * d[3] * d[6] * d[17] - 2 * d[1] * d[2] * d[6] * d[18] - 2 * d[0] * d[3] * d[10] - 2 * d[1] * d[3] * d[11] + 2 * d[2] * d[12] * d[13] + 2 * d[0] * d[3] * d[17] - 2 * d[0] * d[2] * d[18] - 2 * d[3] * d[12];
					coeffs[13] = -d[3] * d[4] * d[9] * d[14] + d[2] * d[4] * d[9] * d[15] - d[3] * d[4] * d[12] * d[17] + d[2] * d[4] * d[12] * d[18];
					coeffs[14] = d[1] * d[2] * d[6] * d[13] - d[0] * d[3] * d[7] * d[14] - d[1] * d[3] * d[8] * d[14] + d[0] * d[2] * d[7] * d[15] + d[1] * d[2] * d[8] * d[15] - d[0] * d[3] * d[10] * d[17] - d[1] * d[3] * d[11] * d[17] + d[0] * d[2] * d[10] * d[18] + d[1] * d[2] * d[11] * d[18] - d[1] * d[3] * d[6] + d[0] * d[2] * d[13] - d[3] * d[9] * d[14] + d[2] * d[9] * d[15] - d[3] * d[12] * d[17] + d[2] * d[12] * d[18] - d[0] * d[3];

					static const int coeffs_ind[] = {0, 9, 1, 9, 10, 3, 10, 2, 11, 4, 11, 12, 5, 13, 6, 12, 7, 13, 14, 8, 14};

					static const int C_ind[] = {0, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 25};

					MatrixXd C = MatrixXd::Zero(3, 9);
					for (int i = 0; i < 21; i++)
					{
						C(C_ind[i]) = coeffs(coeffs_ind[i]);
					}

					MatrixXd C0 = C.leftCols(3);
					MatrixXd C1 = C.rightCols(6);
					MatrixXd C12 = C0.fullPivLu().solve(C1);
					MatrixXd RR(9, 6);
					RR << -C12.bottomRows(3), MatrixXd::Identity(6, 6);

					static const int AM_ind[] = {0, 1, 3, 2, 4, 6};
					MatrixXd AM(6, 6);
					for (int i = 0; i < 6; i++)
					{
						AM.row(i) = RR.row(AM_ind[i]);
					}

					EigenSolver<MatrixXd> es(AM);
					ArrayXcd D = es.eigenvalues();
					ArrayXXcd V = es.eigenvectors();
					V = (V / V.row(5).replicate(6, 1)).eval();

					MatrixXcd sols(2, 6);
					sols.row(0) = D.transpose();
					sols.row(1) = V.row(4);

					for (int i = 0; i < 6; i++)
					{
						if (sols(0, i).imag() > std::numeric_limits<double>::epsilon() || sols(0, i).real() > 1 || sols(0, i).real() < -1) // cos(theta) 1-s^2 > 0
							continue;

						double r = sols(0, i).real();
						double rsqr = r * r;

						double dis = sols(1, i).real(); // distortion

						Eigen::Matrix<double, 3, 3> Ry;
						Ry << (1 - rsqr), 0, 2 * r,
							0, 1 + rsqr, 0,
							-2 * r, 0, (1 - rsqr);
						Ry = Ry / (1 + rsqr);

						Eigen::Matrix<double, 3, 3> H;

						H = gravity_destination.transpose() * Ry * gravity_source;

						if (H.hasNaN())
							continue;

						RadialHomography model;
						model.descriptor.block<3, 3>(0, 0) = H;
						model.descriptor(0, 3) = dis;
						model.descriptor(1, 3) = dis;
						models_.push_back(model);
					}
				}

				else
				{
					double r1 = u1 * u1 + v1 * v1,
						   r2 = u2 * u2 + v2 * v2,
						   k0 = r1 * u1 * v2 ^ 2 + r2 * u2 * v1 ^ 2 - r1 * u2 * v1 * v2 - r2 * u1 * v1 * v2,
						   k1 = 2 * r1 * v1 * v2 - 2 * r2 * v1 * v2,
						   k2 = r1 * u1 * v2 ^ 2 + r2 * u2 * v1 ^ 2 + r1 * u2 * v1 * v2 + r2 * u1 * v1 * v2,
						   c0 = u2 * v1 - u1 * v2,
						   c1 = -2 * v2,
						   c2 = u1 * v2 + u2 * v1,
						   c3 = 2 * r1 * v2;

					std::complex<double> roots[2];
					solve_quadratic(k2, k1, k0, roots); // r

					for (size_t k = 0; k < 2; ++k)
					{
						if (roots[k].imag() > std::numeric_limits<double>::epsilon() || roots[k].real() > 1 || roots[k].real() < -1) //
							continue;

						double r = roots[k].real();
						double rsqr = r*r;

						double dis = (c0 + c1 * r + c2 * rsqr) / (c3 * r); // distortion
						if (dis > 0)
							continue;

						Eigen::Matrix<double, 3, 3> Ry;
						Ry << (1 - rsqr), 0, 2 * r,
							0, 1 + rsqr, 0,
							-2 * r, 0, (1 - rsqr);

						Eigen::Matrix<double, 3, 3> H;
						H = gravity_destination.transpose() * Ry * gravity_source;

						if (H.hasNaN())
							continue;

						RadialHomography model;
						model.descriptor.block<3, 3>(0, 0) = H;
						model.descriptor(0, 3) = dis;
						model.descriptor(1, 3) = dis;
						models_.push_back(model);
					}
				}

				return models_.size() > 0;

			} // namespace solver
		}	  // namespace solver
	}		  // namespace estimator
} // namespace gcransac
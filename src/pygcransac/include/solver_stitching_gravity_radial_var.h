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
			class StitchingGravityRadialVar : public SolverEngine
			{
			public:
				// StitchingGravityRadialVar() : gravity_source(Eigen::Matrix3d::Identity()),
				// 								gravity_destination(Eigen::Matrix3d::Identity())
				// {
				// }

				~StitchingGravityRadialVar()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
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
					return "Gravity Var Radial (3PC)";
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

			void StitchingGravityRadialVar::solve_quadratic(double a, double b, double c, std::complex<double> roots[2]) const
			{

				std::complex<double> b2m4ac = b * b - 4 * a * c;
				std::complex<double> sq = std::sqrt(b2m4ac);

				// Choose sign to avoid cancellations
				roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
				roots[1] = c / (a * roots[0]);
			}

			OLGA_INLINE bool StitchingGravityRadialVar::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;

				if (sample_number_ < sampleSize() ||
					sample_number_ > sampleSize())
					return false;

				// Building the coefficient matrices
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t cols = data_.cols;
				const size_t pointNumber = sampleSize(); // 3

				const double
					&u1 = data_.at<double>(sample_[0], 0),
					&v1 = data_.at<double>(sample_[0], 1),
					&u2 = data_.at<double>(sample_[0], 2),
					&v2 = data_.at<double>(sample_[0], 3),
					&u3 = data_.at<double>(sample_[1], 0),
					&v3 = data_.at<double>(sample_[1], 1),
					&u4 = data_.at<double>(sample_[1], 2),
					&v4 = data_.at<double>(sample_[1], 3),
					&u5 = data_.at<double>(sample_[2], 0),
					&v5 = data_.at<double>(sample_[2], 1),
					&u6 = data_.at<double>(sample_[2], 2),
					&v6 = data_.at<double>(sample_[2], 3);

                double r1 = u1 * u1 + v1 * v1,
                    r2 = u2 * u2 + v2 * v2,
                    r3 = u3 * u3 + v3 * v3,
                    r4 = u4 * u4 + v4 * v4,
                    r5 = u5 * u5 + v5 * v5,
                    r6 = u6 * u6 + v6 * v6;

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

                VectorXd d(29);
                d << u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6, r1, r3, r5, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7;
                VectorXd coeffs(27);
                coeffs[0] = -d[3] * d[12] * d[18] * d[23] + d[2] * d[12] * d[18] * d[24] + d[3] * d[12] * d[21] * d[26] - d[2] * d[12] * d[21] * d[27];
                coeffs[1] = -d[3] * d[18] * d[23] + d[2] * d[18] * d[24] + d[3] * d[21] * d[26] - d[2] * d[21] * d[27];
                coeffs[2] = 2 * d[2] * d[12] * d[21] * d[22] - 2 * d[3] * d[12] * d[21];
                coeffs[3] = -d[1] * d[2] * d[15] * d[22] - d[0] * d[3] * d[16] * d[23] - d[1] * d[3] * d[17] * d[23] + d[0] * d[2] * d[16] * d[24] + d[1] * d[2] * d[17] * d[24] + d[0] * d[3] * d[19] * d[26] + d[1] * d[3] * d[20] * d[26] - d[0] * d[2] * d[19] * d[27] - d[1] * d[2] * d[20] * d[27] + d[1] * d[3] * d[15] - d[0] * d[2] * d[22] + d[0] * d[3];
                coeffs[4] = 2 * d[2] * d[21] * d[22] - 2 * d[3] * d[21];
                coeffs[5] = -d[3] * d[12] * d[18] * d[23] + d[2] * d[12] * d[18] * d[24] - d[3] * d[12] * d[21] * d[26] + d[2] * d[12] * d[21] * d[27];
                coeffs[6] = 2 * d[0] * d[2] * d[19] * d[22] + 2 * d[1] * d[2] * d[20] * d[22] + 2 * d[1] * d[3] * d[15] * d[26] - 2 * d[1] * d[2] * d[15] * d[27] - 2 * d[0] * d[3] * d[19] - 2 * d[1] * d[3] * d[20] + 2 * d[0] * d[3] * d[26] - 2 * d[0] * d[2] * d[27];
                coeffs[7] = -d[3] * d[18] * d[23] + d[2] * d[18] * d[24] - d[3] * d[21] * d[26] + d[2] * d[21] * d[27];
                coeffs[8] = d[1] * d[2] * d[15] * d[22] - d[0] * d[3] * d[16] * d[23] - d[1] * d[3] * d[17] * d[23] + d[0] * d[2] * d[16] * d[24] + d[1] * d[2] * d[17] * d[24] - d[0] * d[3] * d[19] * d[26] - d[1] * d[3] * d[20] * d[26] + d[0] * d[2] * d[19] * d[27] + d[1] * d[2] * d[20] * d[27] - d[1] * d[3] * d[15] + d[0] * d[2] * d[22] - d[0] * d[3];
                coeffs[9] = -d[7] * d[13] * d[18] * d[23] + d[6] * d[13] * d[18] * d[24] + d[7] * d[13] * d[21] * d[26] - d[6] * d[13] * d[21] * d[27];
                coeffs[10] = -d[7] * d[18] * d[23] + d[6] * d[18] * d[24] + d[7] * d[21] * d[26] - d[6] * d[21] * d[27];
                coeffs[11] = 2 * d[6] * d[13] * d[21] * d[22] - 2 * d[7] * d[13] * d[21];
                coeffs[12] = -d[5] * d[6] * d[15] * d[22] - d[4] * d[7] * d[16] * d[23] - d[5] * d[7] * d[17] * d[23] + d[4] * d[6] * d[16] * d[24] + d[5] * d[6] * d[17] * d[24] + d[4] * d[7] * d[19] * d[26] + d[5] * d[7] * d[20] * d[26] - d[4] * d[6] * d[19] * d[27] - d[5] * d[6] * d[20] * d[27] + d[5] * d[7] * d[15] - d[4] * d[6] * d[22] + d[4] * d[7];
                coeffs[13] = 2 * d[6] * d[21] * d[22] - 2 * d[7] * d[21];
                coeffs[14] = -d[7] * d[13] * d[18] * d[23] + d[6] * d[13] * d[18] * d[24] - d[7] * d[13] * d[21] * d[26] + d[6] * d[13] * d[21] * d[27];
                coeffs[15] = 2 * d[4] * d[6] * d[19] * d[22] + 2 * d[5] * d[6] * d[20] * d[22] + 2 * d[5] * d[7] * d[15] * d[26] - 2 * d[5] * d[6] * d[15] * d[27] - 2 * d[4] * d[7] * d[19] - 2 * d[5] * d[7] * d[20] + 2 * d[4] * d[7] * d[26] - 2 * d[4] * d[6] * d[27];
                coeffs[16] = -d[7] * d[18] * d[23] + d[6] * d[18] * d[24] - d[7] * d[21] * d[26] + d[6] * d[21] * d[27];
                coeffs[17] = d[5] * d[6] * d[15] * d[22] - d[4] * d[7] * d[16] * d[23] - d[5] * d[7] * d[17] * d[23] + d[4] * d[6] * d[16] * d[24] + d[5] * d[6] * d[17] * d[24] - d[4] * d[7] * d[19] * d[26] - d[5] * d[7] * d[20] * d[26] + d[4] * d[6] * d[19] * d[27] + d[5] * d[6] * d[20] * d[27] - d[5] * d[7] * d[15] + d[4] * d[6] * d[22] - d[4] * d[7];
                coeffs[18] = -d[11] * d[14] * d[18] * d[23] + d[10] * d[14] * d[18] * d[24] + d[11] * d[14] * d[21] * d[26] - d[10] * d[14] * d[21] * d[27];
                coeffs[19] = -d[11] * d[18] * d[23] + d[10] * d[18] * d[24] + d[11] * d[21] * d[26] - d[10] * d[21] * d[27];
                coeffs[20] = 2 * d[10] * d[14] * d[21] * d[22] - 2 * d[11] * d[14] * d[21];
                coeffs[21] = -d[9] * d[10] * d[15] * d[22] - d[8] * d[11] * d[16] * d[23] - d[9] * d[11] * d[17] * d[23] + d[8] * d[10] * d[16] * d[24] + d[9] * d[10] * d[17] * d[24] + d[8] * d[11] * d[19] * d[26] + d[9] * d[11] * d[20] * d[26] - d[8] * d[10] * d[19] * d[27] - d[9] * d[10] * d[20] * d[27] + d[9] * d[11] * d[15] - d[8] * d[10] * d[22] + d[8] * d[11];
                coeffs[22] = 2 * d[10] * d[21] * d[22] - 2 * d[11] * d[21];
                coeffs[23] = -d[11] * d[14] * d[18] * d[23] + d[10] * d[14] * d[18] * d[24] - d[11] * d[14] * d[21] * d[26] + d[10] * d[14] * d[21] * d[27];
                coeffs[24] = 2 * d[8] * d[10] * d[19] * d[22] + 2 * d[9] * d[10] * d[20] * d[22] + 2 * d[9] * d[11] * d[15] * d[26] - 2 * d[9] * d[10] * d[15] * d[27] - 2 * d[8] * d[11] * d[19] - 2 * d[9] * d[11] * d[20] + 2 * d[8] * d[11] * d[26] - 2 * d[8] * d[10] * d[27];
                coeffs[25] = -d[11] * d[18] * d[23] + d[10] * d[18] * d[24] - d[11] * d[21] * d[26] + d[10] * d[21] * d[27];
                coeffs[26] = d[9] * d[10] * d[15] * d[22] - d[8] * d[11] * d[16] * d[23] - d[9] * d[11] * d[17] * d[23] + d[8] * d[10] * d[16] * d[24] + d[9] * d[10] * d[17] * d[24] - d[8] * d[11] * d[19] * d[26] - d[9] * d[11] * d[20] * d[26] + d[8] * d[10] * d[19] * d[27] + d[9] * d[10] * d[20] * d[27] - d[9] * d[11] * d[15] + d[8] * d[10] * d[22] - d[8] * d[11];

                static const int coeffs_ind[] = { 0, 9, 18, 1, 10, 19, 3, 12, 21, 2, 11, 20, 4, 13, 22, 5, 14, 23, 6, 15, 24, 7, 16, 25, 8, 17, 26 };

                static const int C_ind[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 };

                MatrixXd C = MatrixXd::Zero(3, 9);
                for (int i = 0; i < 27; i++)
                {
                    C(C_ind[i]) = coeffs(coeffs_ind[i]);
                }

                MatrixXd C0 = C.leftCols(3);
                MatrixXd C1 = C.rightCols(6);
                MatrixXd C12 = C0.fullPivLu().solve(C1);
                MatrixXd RR(9, 6);
                RR << -C12.bottomRows(3), MatrixXd::Identity(6, 6);

                static const int AM_ind[] = { 0, 1, 3, 2, 4, 6 };
                MatrixXd AM(6, 6);
                for (int i = 0; i < 6; i++)
                {
                    AM.row(i) = RR.row(AM_ind[i]);
                }

                EigenSolver<MatrixXd> es(AM);
                ArrayXcd D = es.eigenvalues();
                ArrayXXcd V = es.eigenvectors();
                V = (V / V.row(5).replicate(6, 1)).eval();

                MatrixXcd sols(3, 6);
                sols.row(0) = D.transpose();
                sols.row(1) = V.row(4);
                sols.row(2) = V.row(0) / (sols.row(0).array() * sols.row(1).array());

                for (int k = 0; k < 6; ++k)
                {
                    if (sols(0, k).imag() > std::numeric_limits<double>::epsilon() || sols(0, k).real() > 1 || sols(0, k).real() < -1 || sols(1, k).real() < 0) // cos(theta) 1-s^2 > 0
                        continue;

                    double r = sols(0, k).real();
                    double rsqr = r * r;
                    double focal_a = sols(1, k).real();
                    if (focal_a < 0)
                        continue;

                    double dis_a = sols(2, k).real();

                    Eigen::Matrix<double, 3, 3> Ry;
                    Ry << (1 - rsqr), 0, 2 * r,
                        0, 1 + rsqr, 0,
                        -2 * r, 0, (1 - rsqr);

                    Eigen::Vector3d m1(u1, v1, focal_a * (1 + dis_a * r1));
                    Eigen::Vector3d m11 = gravity_destination.transpose() * Ry * gravity_source * m1;

                    Eigen::Vector3d m3(u3, v3, focal_a * (1 + dis_a * r3));
                    Eigen::Vector3d m33 = gravity_destination.transpose() * Ry * gravity_source * m3;

                    double co1 = m33(1) * v2 * m11(2);
                    double co2 = m11(1) * v4 * m33(2);
                    double dis_b = (co2 - co1) / (co1 * r4 - co2 * r2);
                    double focal_b = v2 * m11(2) / (m11(1) * (1 + r2 * dis_b));
                    if (focal_b < 0)
                        continue;

                    Eigen::Matrix<double, 3, 3> K2;
                    K2 << focal_b, 0, 0,
                        0, focal_b, 0,
                        0, 0, 1.0;

                    Eigen::Matrix<double, 3, 3> K1inv;
                    K1inv << 1.0 / focal_a, 0, 0,
                        0, 1.0 / focal_a, 0,
                        0, 0, 1.0;

                    Eigen::Matrix<double, 3, 3> H;
					H = K2 * gravity_destination.transpose() * Ry * gravity_source * K1inv;

                    if (H.hasNaN())
                        continue;

					RadialHomography model;
					model.descriptor.block<3, 3>(0, 0) = H;
					model.descriptor(0, 3) = dis_a;
					model.descriptor(1, 3) = dis_b;
					model.descriptor(2, 3) = sqrt(focal_a * focal_b);
					models_.push_back(model);
				}

				return models_.size() > 0;
			} // namespace solver

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac

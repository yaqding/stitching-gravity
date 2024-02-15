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
#include "charpoly.h"
#include "Polynomial.hpp"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class StitchingGravityRadialEqual : public SolverEngine
			{
			public:
				// StitchingGravityRadialEqual() : gravity_source(Eigen::Matrix3d::Identity()),
				// 								gravity_destination(Eigen::Matrix3d::Identity())
				// {
				// }

				~StitchingGravityRadialEqual()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
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

			void StitchingGravityRadialEqual::solve_quadratic(double a, double b, double c, std::complex<double> roots[2]) const
			{
				std::complex<double> b2m4ac = b * b - 4 * a * c;
				std::complex<double> sq = std::sqrt(b2m4ac);

				// Choose sign to avoid cancellations
				roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
				roots[1] = c / (a * roots[0]);
			}

			OLGA_INLINE bool StitchingGravityRadialEqual::estimateModel(
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
					&u3 = data_.at<double>(sample_[1], 0),
					&v3 = data_.at<double>(sample_[1], 1),
					&u4 = data_.at<double>(sample_[1], 2),
					&v4 = data_.at<double>(sample_[1], 3);

				if (gravity_source(0, 0) != 1)
				{
					double r1 = u1 * u1 + v1 * v1,
						   r2 = u2 * u2 + v2 * v2,
						   r3 = u3 * u3 + v3 * v3,
						   r4 = u4 * u4 + v4 * v4;

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

					double d[26] = {u1, v1, u2, v2, u3, v3, u4, v4, r1, r2, r3, r4, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7};

					double c1 = -d[8] * d[15] * d[21] - d[9] * d[15] * d[21] + d[8] * d[18] * d[24] + d[9] * d[18] * d[24];
					double c2 = -d[8] * d[9] * d[15] * d[21] + d[8] * d[9] * d[18] * d[24];
					double c3 = d[1] * d[12] * d[19] - d[0] * d[13] * d[21] - d[1] * d[14] * d[21] + d[3] * d[15] * d[22] + d[0] * d[16] * d[24] + d[1] * d[17] * d[24] - d[3] * d[18] * d[25] + d[0] * d[19];
					double c5 = d[1] * d[9] * d[12] * d[19] - d[0] * d[9] * d[13] * d[21] - d[1] * d[9] * d[14] * d[21] + d[3] * d[8] * d[15] * d[22] + d[0] * d[9] * d[16] * d[24] + d[1] * d[9] * d[17] * d[24] - d[3] * d[8] * d[18] * d[25] + d[0] * d[9] * d[19];
					double c6 = -2 * d[8] * d[18] * d[19] - 2 * d[9] * d[18] * d[19];
					double c7 = -2 * d[8] * d[9] * d[18] * d[19];
					double c8 = d[0] * d[3] * d[13] * d[22] + d[1] * d[3] * d[14] * d[22] - d[0] * d[3] * d[16] * d[25] - d[1] * d[3] * d[17] * d[25];
					double c9 = -2 * d[0] * d[16] * d[19] - 2 * d[1] * d[17] * d[19] + 2 * d[1] * d[12] * d[24] + 2 * d[0] * d[24];
					double c11 = -2 * d[0] * d[9] * d[16] * d[19] - 2 * d[1] * d[9] * d[17] * d[19] + 2 * d[1] * d[9] * d[12] * d[24] + 2 * d[0] * d[9] * d[24];
					double c12 = -d[8] * d[15] * d[21] - d[9] * d[15] * d[21] - d[8] * d[18] * d[24] - d[9] * d[18] * d[24];
					double c13 = -d[8] * d[9] * d[15] * d[21] - d[8] * d[9] * d[18] * d[24];
					double c14 = -2 * d[1] * d[3] * d[12] * d[25] - 2 * d[0] * d[3] * d[25];
					double c15 = -d[1] * d[12] * d[19] - d[0] * d[13] * d[21] - d[1] * d[14] * d[21] + d[3] * d[15] * d[22] - d[0] * d[16] * d[24] - d[1] * d[17] * d[24] + d[3] * d[18] * d[25] - d[0] * d[19];
					double c16 = -d[1] * d[9] * d[12] * d[19] - d[0] * d[9] * d[13] * d[21] - d[1] * d[9] * d[14] * d[21] + d[3] * d[8] * d[15] * d[22] - d[0] * d[9] * d[16] * d[24] - d[1] * d[9] * d[17] * d[24] + d[3] * d[8] * d[18] * d[25] - d[0] * d[9] * d[19];
					double c17 = d[0] * d[3] * d[13] * d[22] + d[1] * d[3] * d[14] * d[22] + d[0] * d[3] * d[16] * d[25] + d[1] * d[3] * d[17] * d[25];
					double d0 = -d[3] * d[15] * d[20] + d[2] * d[15] * d[21] + d[3] * d[18] * d[23] - d[2] * d[18] * d[24];
					double d1 = -d[3] * d[8] * d[15] * d[20] + d[2] * d[8] * d[15] * d[21] + d[3] * d[8] * d[18] * d[23] - d[2] * d[8] * d[18] * d[24];
					double d2 = -d[1] * d[2] * d[12] * d[19] - d[0] * d[3] * d[13] * d[20] - d[1] * d[3] * d[14] * d[20] + d[0] * d[2] * d[13] * d[21] + d[1] * d[2] * d[14] * d[21] + d[0] * d[3] * d[16] * d[23] + d[1] * d[3] * d[17] * d[23] - d[0] * d[2] * d[16] * d[24] - d[1] * d[2] * d[17] * d[24] + d[1] * d[3] * d[12] - d[0] * d[2] * d[19] + d[0] * d[3];
					double d3 = 2 * d[2] * d[18] * d[19] - 2 * d[3] * d[18];
					double d4 = 2 * d[2] * d[8] * d[18] * d[19] - 2 * d[3] * d[8] * d[18];
					double d5 = 2 * d[0] * d[2] * d[16] * d[19] + 2 * d[1] * d[2] * d[17] * d[19] + 2 * d[1] * d[3] * d[12] * d[23] - 2 * d[1] * d[2] * d[12] * d[24] - 2 * d[0] * d[3] * d[16] - 2 * d[1] * d[3] * d[17] + 2 * d[0] * d[3] * d[23] - 2 * d[0] * d[2] * d[24];
					double d6 = -d[3] * d[15] * d[20] + d[2] * d[15] * d[21] - d[3] * d[18] * d[23] + d[2] * d[18] * d[24];
					double d7 = -d[3] * d[8] * d[15] * d[20] + d[2] * d[8] * d[15] * d[21] - d[3] * d[8] * d[18] * d[23] + d[2] * d[8] * d[18] * d[24];
					double d8 = d[1] * d[2] * d[12] * d[19] - d[0] * d[3] * d[13] * d[20] - d[1] * d[3] * d[14] * d[20] + d[0] * d[2] * d[13] * d[21] + d[1] * d[2] * d[14] * d[21] - d[0] * d[3] * d[16] * d[23] - d[1] * d[3] * d[17] * d[23] + d[0] * d[2] * d[16] * d[24] + d[1] * d[2] * d[17] * d[24] - d[1] * d[3] * d[12] + d[0] * d[2] * d[19] - d[0] * d[3];
					double e0 = -d[7] * d[15] * d[20] + d[6] * d[15] * d[21] + d[7] * d[18] * d[23] - d[6] * d[18] * d[24];
					double e1 = -d[7] * d[10] * d[15] * d[20] + d[6] * d[10] * d[15] * d[21] + d[7] * d[10] * d[18] * d[23] - d[6] * d[10] * d[18] * d[24];
					double e2 = -d[5] * d[6] * d[12] * d[19] - d[4] * d[7] * d[13] * d[20] - d[5] * d[7] * d[14] * d[20] + d[4] * d[6] * d[13] * d[21] + d[5] * d[6] * d[14] * d[21] + d[4] * d[7] * d[16] * d[23] + d[5] * d[7] * d[17] * d[23] - d[4] * d[6] * d[16] * d[24] - d[5] * d[6] * d[17] * d[24] + d[5] * d[7] * d[12] - d[4] * d[6] * d[19] + d[4] * d[7];
					double e3 = 2 * d[6] * d[18] * d[19] - 2 * d[7] * d[18];
					double e4 = 2 * d[6] * d[10] * d[18] * d[19] - 2 * d[7] * d[10] * d[18];
					double e5 = 2 * d[4] * d[6] * d[16] * d[19] + 2 * d[5] * d[6] * d[17] * d[19] + 2 * d[5] * d[7] * d[12] * d[23] - 2 * d[5] * d[6] * d[12] * d[24] - 2 * d[4] * d[7] * d[16] - 2 * d[5] * d[7] * d[17] + 2 * d[4] * d[7] * d[23] - 2 * d[4] * d[6] * d[24];
					double e6 = -d[7] * d[15] * d[20] + d[6] * d[15] * d[21] - d[7] * d[18] * d[23] + d[6] * d[18] * d[24];
					double e7 = -d[7] * d[10] * d[15] * d[20] + d[6] * d[10] * d[15] * d[21] - d[7] * d[10] * d[18] * d[23] + d[6] * d[10] * d[18] * d[24];
					double e8 = d[5] * d[6] * d[12] * d[19] - d[4] * d[7] * d[13] * d[20] - d[5] * d[7] * d[14] * d[20] + d[4] * d[6] * d[13] * d[21] + d[5] * d[6] * d[14] * d[21] - d[4] * d[7] * d[16] * d[23] - d[5] * d[7] * d[17] * d[23] + d[4] * d[6] * d[16] * d[24] + d[5] * d[6] * d[17] * d[24] - d[5] * d[7] * d[12] + d[4] * d[6] * d[19] - d[4] * d[7];

					double f0 = d8 * e7 - d7 * e8;
					double f1 = d5 * e7 - d4 * e8 - d7 * e5 + d8 * e4;
					double f2 = d2 * e7 - d1 * e8 - d4 * e5 + d5 * e4 - d7 * e2 + d8 * e1;
					double f3 = d2 * e4 - d1 * e5 - d4 * e2 + d5 * e1;
					double f4 = d2 * e1 - d1 * e2;

					double f5 = d7 * e6 - d6 * e7;
					double f6 = d4 * e6 - d3 * e7 - d6 * e4 + d7 * e3;
					double f7 = d1 * e6 - d0 * e7 - d3 * e4 + d4 * e3 - d6 * e1 + d7 * e0;
					double f8 = d1 * e3 - d0 * e4 - d3 * e1 + d4 * e0;
					double f9 = d1 * e0 - d0 * e1;

					double g0 = c2 * d2 * d2 - c2 * d5 * d5 + c8 * d1 * d1 + c2 * d8 * d8 - c8 * d4 * d4 + c8 * d7 * d7 - c13 * d2 * d2 + c13 * d5 * d5 - c17 * d1 * d1 + c17 * d4 * d4 - c5 * d1 * d2 - 2 * c2 * d2 * d8 + c5 * d1 * d8 + c5 * d2 * d7 + c5 * d4 * d5 - 2 * c7 * d2 * d5 - 2 * c8 * d1 * d7 + c11 * d1 * d5 + c11 * d2 * d4 - 2 * c14 * d1 * d4 + c16 * d1 * d2 - c5 * d7 * d8 + 2 * c7 * d5 * d8 - c11 * d4 * d8 - c11 * d5 * d7 + 2 * c13 * d2 * d8 + 2 * c14 * d4 * d7 - c16 * d1 * d8 - c16 * d2 * d7 - c16 * d4 * d5 + 2 * c17 * d1 * d7;
					double g1 = c7 * d5 * d5 - c7 * d2 * d2 - c14 * d1 * d1 + c14 * d4 * d4 - 2 * c2 * d2 * d5 + c5 * d1 * d5 + c5 * d2 * d4 - 2 * c8 * d1 * d4 + c11 * d1 * d2 + 2 * c2 * d5 * d8 - c5 * d4 * d8 - c5 * d5 * d7 + 2 * c7 * d2 * d8 + 2 * c8 * d4 * d7 - c11 * d1 * d8 - c11 * d2 * d7 - c11 * d4 * d5 + 2 * c13 * d2 * d5 + 2 * c14 * d1 * d7 - c16 * d1 * d5 - c16 * d2 * d4 + 2 * c17 * d1 * d4;
					double g2 = c2 * d5 * d5 - c2 * d2 * d2 - c8 * d1 * d1 + c8 * d4 * d4 + c13 * d2 * d2 + c17 * d1 * d1 + c5 * d1 * d2 + 2 * c2 * d2 * d8 - c5 * d1 * d8 - c5 * d2 * d7 - c5 * d4 * d5 + 2 * c7 * d2 * d5 + 2 * c8 * d1 * d7 - c11 * d1 * d5 - c11 * d2 * d4 + 2 * c14 * d1 * d4 - c16 * d1 * d2;
					double g3 = c7 * d2 * d2 + c14 * d1 * d1 + 2 * c2 * d2 * d5 - c5 * d1 * d5 - c5 * d2 * d4 + 2 * c8 * d1 * d4 - c11 * d1 * d2;
					double g4 = c8 * d1 * d1 - c5 * d1 * d2 + c2 * d2 * d2;

					double g5 = c3 * d1 * d1 - c3 * d4 * d4 + c3 * d7 * d7 - c15 * d1 * d1 + c15 * d4 * d4 - c1 * d1 * d2 + 2 * c2 * d0 * d2 - c5 * d0 * d1 + c1 * d1 * d8 + c1 * d2 * d7 + c1 * d4 * d5 - 2 * c2 * d0 * d8 - 2 * c2 * d2 * d6 - 2 * c2 * d3 * d5 - 2 * c3 * d1 * d7 + c5 * d0 * d7 + c5 * d1 * d6 + c5 * d3 * d4 + c6 * d1 * d5 + c6 * d2 * d4 - 2 * c7 * d0 * d5 - 2 * c7 * d2 * d3 - 2 * c9 * d1 * d4 + c11 * d0 * d4 + c11 * d1 * d3 + c12 * d1 * d2 - 2 * c13 * d0 * d2 - c1 * d7 * d8 + 2 * c2 * d6 * d8 + c16 * d0 * d1 - c5 * d6 * d7 - c6 * d4 * d8 - c6 * d5 * d7 + 2 * c7 * d3 * d8 + 2 * c7 * d5 * d6 + 2 * c9 * d4 * d7 - c11 * d3 * d7 - c11 * d4 * d6 - c12 * d1 * d8 - c12 * d2 * d7 - c12 * d4 * d5 + 2 * c13 * d0 * d8 + 2 * c13 * d2 * d6 + 2 * c13 * d3 * d5 + 2 * c15 * d1 * d7 - c16 * d0 * d7 - c16 * d1 * d6 - c16 * d3 * d4;
					double g6 = c9 * d4 * d4 - c9 * d1 * d1 + c1 * d1 * d5 + c1 * d2 * d4 - 2 * c2 * d0 * d5 - 2 * c2 * d2 * d3 - 2 * c3 * d1 * d4 + c5 * d0 * d4 + c5 * d1 * d3 + c6 * d1 * d2 - 2 * c7 * d0 * d2 + c11 * d0 * d1 - c1 * d4 * d8 - c1 * d5 * d7 + 2 * c2 * d3 * d8 + 2 * c2 * d5 * d6 + 2 * c3 * d4 * d7 - c5 * d3 * d7 - c5 * d4 * d6 - c6 * d1 * d8 - c6 * d2 * d7 - c6 * d4 * d5 + 2 * c7 * d0 * d8 + 2 * c7 * d2 * d6 + 2 * c7 * d3 * d5 + 2 * c9 * d1 * d7 - c11 * d0 * d7 - c11 * d1 * d6 - c11 * d3 * d4 - c12 * d1 * d5 - c12 * d2 * d4 + 2 * c13 * d0 * d5 + 2 * c13 * d2 * d3 + 2 * c15 * d1 * d4 - c16 * d0 * d4 - c16 * d1 * d3;
					double g7 = c3 * d4 * d4 - c3 * d1 * d1 + c15 * d1 * d1 + c1 * d1 * d2 - 2 * c2 * d0 * d2 + c5 * d0 * d1 - c1 * d1 * d8 - c1 * d2 * d7 - c1 * d4 * d5 + 2 * c2 * d0 * d8 + 2 * c2 * d2 * d6 + 2 * c2 * d3 * d5 + 2 * c3 * d1 * d7 - c5 * d0 * d7 - c5 * d1 * d6 - c5 * d3 * d4 - c6 * d1 * d5 - c6 * d2 * d4 + 2 * c7 * d0 * d5 + 2 * c7 * d2 * d3 + 2 * c9 * d1 * d4 - c11 * d0 * d4 - c11 * d1 * d3 - c12 * d1 * d2 + 2 * c13 * d0 * d2 - c16 * d0 * d1;
					double g8 = c9 * d1 * d1 - c1 * d1 * d5 - c1 * d2 * d4 + 2 * c2 * d0 * d5 + 2 * c2 * d2 * d3 + 2 * c3 * d1 * d4 - c5 * d0 * d4 - c5 * d1 * d3 - c6 * d1 * d2 + 2 * c7 * d0 * d2 - c11 * d0 * d1;
					double g9 = c3 * d1 * d1 - c1 * d1 * d2 + 2 * c2 * d0 * d2 - c5 * d0 * d1;

					double h0 = f0 * g5 + f5 * g0;
					double h1 = f0 * g6 + f1 * g5 + f5 * g1 + f6 * g0;
					double h2 = f0 * g7 + f1 * g6 + f2 * g5 + f5 * g2 + f6 * g1 + f7 * g0;
					double h3 = f0 * g8 + f1 * g7 + f2 * g6 + f3 * g5 + f5 * g3 + f6 * g2 + f7 * g1 + f8 * g0;
					double h4 = f0 * g9 + f1 * g8 + f2 * g7 + f3 * g6 + f4 * g5 + f5 * g4 + f6 * g3 + f7 * g2 + f8 * g1 + f9 * g0;
					double h5 = f1 * g9 + f2 * g8 + f3 * g7 + f4 * g6 + f6 * g4 + f7 * g3 + f8 * g2 + f9 * g1;
					double h6 = f2 * g9 + f3 * g8 + f4 * g7 + f7 * g4 + f8 * g3 + f9 * g2;
					double h7 = f3 * g9 + f4 * g8 + f8 * g4 + f9 * g3;
					double h8 = f4 * g9 + f9 * g4;

					using polynomial::Polynomial;

					Matrix<double, 9, 1> hcoeffs;
					hcoeffs << h8,
						h7,
						h6,
						h5,
						h4,
						h3,
						h2,
						h1,
						h0;
					Polynomial<8> h(hcoeffs);

					std::vector<double> roots;
					h.realRootsSturm(-1, 1, roots);

					if (roots.size() > 0 && roots.size() <= 8)
					{
						for (int i = 0; i < roots.size(); i++)
						{
							if (roots[i] > 1 || roots[i] < -1) // cos(theta) 1-s^2 > 0
								continue;

							double r = roots[i];
							double rsqr = r * r;

							double co1 = d0 * rsqr + d3 * r + d6;
							double co2 = d1 * rsqr + d4 * r + d7;
							double co3 = d2 * rsqr + d5 * r + d8;

							double co4 = e0 * rsqr + e3 * r + e6;
							double co5 = e1 * rsqr + e4 * r + e7;
							double co6 = e2 * rsqr + e5 * r + e8;

							double focal = (co2 * co6 - co3 * co5) / (co1 * co5 - co2 * co4); // focal length
							if (focal < 0)
								continue;

							double dis = (co3 * co4 - co1 * co6) / (co1 * co5 - co2 * co4); // distortion

							Eigen::Matrix<double, 3, 3> Ry;
							Ry << (1 - rsqr), 0, 2 * r,
								0, 1 + rsqr, 0,
								-2 * r, 0, (1 - rsqr);
							Ry = Ry / (1 + rsqr);

							Eigen::Matrix<double, 3, 3> K2;
							K2 << focal, 0, 0,
								0, focal, 0,
								0, 0, 1.0;

							Eigen::Matrix<double, 3, 3> K1inv;
							K1inv << 1.0 / focal, 0, 0,
								0, 1.0 / focal, 0,
								0, 0, 1.0;

							Eigen::Matrix<double, 3, 3> H;

							H = K2 * gravity_destination.transpose() * Ry * gravity_source * K1inv;

							if (H.hasNaN())
								continue;

							RadialHomography model;
							model.descriptor.block<3, 3>(0, 0) = H;
							model.descriptor(0, 3) = dis;
							model.descriptor(1, 3) = dis;
							model.descriptor(2, 3) = focal;
							models_.push_back(model);
						}
					}
				}

				else
				{
					double r1 = u1 * u1 + v1 * v1,
						   r2 = u2 * u2 + v2 * v2,
						   r3 = u3 * u3 + v3 * v3,
						   r4 = u4 * u4 + v4 * v4,
						   c0 = v2 - v1,
						   c1 = -v1 - v2,
						   c2 = -2.0 * u1 * v2,
						   c3 = r1 * v2 - r2 * v1,
						   c4 = -r1 * v2 - r2 * v1,
						   d0 = u2 * v1 - u1 * v2,
						   d1 = u1 * v2 + u2 * v1,
						   d2 = -2.0 * v2,
						   d3 = d2 * r1,
						   d4 = c3 * d1 + c4 * d0 - c2 * d3 - c4 * d1,
						   d5 = c4 * d1,
						   d6 = -c1 * d3 + c4 * d2,
						   e0 = u4 * v3 - u3 * v4,
						   e1 = u3 * v4 + u4 * v3,
						   e2 = -2.0 * v4,
						   e3 = e2 * r3,
						   e4 = c3 * e0,
						   e5 = c3 * e1 + c4 * e0 - c2 * e3,
						   e6 = c4 * e1,
						   e7 = -c0 * e3 + c3 * e2,
						   e8 = -c1 * e3 + c4 * e2,
						   k0 = d6 * e4 - d4 * e7,
						   k1 = d6 * e5 - d4 * e8 - d5 * e7,
						   k2 = d6 * e6 - d5 * e8;

					std::complex<double> roots[2];
					solve_quadratic(k2, k1, k0, roots); // rsqr

					for (size_t k = 0; k < 2; ++k)
					{
						if (roots[k].imag() > std::numeric_limits<double>::epsilon() || roots[k].real() > 1 || roots[k].real() < 0) // 0<r^2<1
							continue;

						double rsqr = roots[k].real();

						for (int i = 0; i < 2; ++i)
						{
							double r = sqrt(rsqr) * std::pow(-1, i);
							double f = -(d4 + d5 * rsqr) / (d6 * r);
							if (f < 0)
								continue;

							double dis = -(c0 * f + c1 * f * rsqr + c2 * r) / (f * (c3 + c4 * rsqr)); // distortion
							if (dis > 0)
								continue;

							Eigen::Matrix<double, 3, 3> Ry;
							Ry << (1 - rsqr), 0, 2 * r,
								0, 1 + rsqr, 0,
								-2 * r, 0, (1 - rsqr);

							Eigen::Matrix<double, 3, 3> K2;
							K2 << f, 0, 0,
								0, f, 0,
								0, 0, 1.0;

							Eigen::Matrix<double, 3, 3> K1inv;
							K1inv << 1.0 / f, 0, 0,
								0, 1.0 / f, 0,
								0, 0, 1.0;

							Eigen::Matrix<double, 3, 3> H;
							H = K2 * gravity_destination.transpose() * Ry * gravity_source * K1inv;

							if (H.hasNaN())
								continue;

							RadialHomography model;
							model.descriptor.block<3, 3>(0, 0) = H;
							model.descriptor(0, 3) = dis;
							model.descriptor(1, 3) = dis;
							model.descriptor(2, 3) = f;
							models_.push_back(model);
						}
					}
				}

				return models_.size() > 0;

			} // namespace solver
		}	  // namespace solver
	}		  // namespace estimator
} // namespace gcransac
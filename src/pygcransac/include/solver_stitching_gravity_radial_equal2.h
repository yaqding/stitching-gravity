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
					VectorXd coeffs(36);
					coeffs[0] = std::pow(d[0], 2) * std::pow(d[2], 2) * d[9] * d[15] + std::pow(d[1], 2) * std::pow(d[2], 2) * d[9] * d[15] + std::pow(d[0], 2) * std::pow(d[3], 2) * d[9] * d[15] + std::pow(d[1], 2) * std::pow(d[3], 2) * d[9] * d[15] - std::pow(d[0], 2) * std::pow(d[2], 2) * d[12] * d[18] - std::pow(d[1], 2) * std::pow(d[2], 2) * d[12] * d[18] - std::pow(d[0], 2) * std::pow(d[3], 2) * d[12] * d[18] - std::pow(d[1], 2) * std::pow(d[3], 2) * d[12] * d[18] + std::pow(d[0], 2) * std::pow(d[2], 2) * d[14] * d[20] + std::pow(d[1], 2) * std::pow(d[2], 2) * d[14] * d[20] + std::pow(d[0], 2) * std::pow(d[3], 2) * d[14] * d[20] + std::pow(d[1], 2) * std::pow(d[3], 2) * d[14] * d[20];
					coeffs[1] = std::pow(d[0], 2) * d[9] * d[15] + std::pow(d[1], 2) * d[9] * d[15] + std::pow(d[2], 2) * d[9] * d[15] + std::pow(d[3], 2) * d[9] * d[15] - std::pow(d[0], 2) * d[12] * d[18] - std::pow(d[1], 2) * d[12] * d[18] - std::pow(d[2], 2) * d[12] * d[18] - std::pow(d[3], 2) * d[12] * d[18] + std::pow(d[0], 2) * d[14] * d[20] + std::pow(d[1], 2) * d[14] * d[20] + std::pow(d[2], 2) * d[14] * d[20] + std::pow(d[3], 2) * d[14] * d[20];
					coeffs[2] = -2 * std::pow(d[0], 2) * std::pow(d[2], 2) * d[14] * d[15] - 2 * std::pow(d[1], 2) * std::pow(d[2], 2) * d[14] * d[15] - 2 * std::pow(d[0], 2) * std::pow(d[3], 2) * d[14] * d[15] - 2 * std::pow(d[1], 2) * std::pow(d[3], 2) * d[14] * d[15] + 2 * std::pow(d[0], 2) * std::pow(d[2], 2) * d[9] * d[20] + 2 * std::pow(d[1], 2) * std::pow(d[2], 2) * d[9] * d[20] + 2 * std::pow(d[0], 2) * std::pow(d[3], 2) * d[9] * d[20] + 2 * std::pow(d[1], 2) * std::pow(d[3], 2) * d[9] * d[20];
					coeffs[3] = d[9] * d[15] - d[12] * d[18] + d[14] * d[20];
					coeffs[4] = d[1] * std::pow(d[2], 2) * d[8] * d[15] + d[1] * std::pow(d[3], 2) * d[8] * d[15] - std::pow(d[0], 2) * d[3] * d[9] * d[16] - std::pow(d[1], 2) * d[3] * d[9] * d[16] - d[0] * std::pow(d[2], 2) * d[10] * d[18] - d[0] * std::pow(d[3], 2) * d[10] * d[18] - d[1] * std::pow(d[2], 2) * d[11] * d[18] - d[1] * std::pow(d[3], 2) * d[11] * d[18] + std::pow(d[0], 2) * d[3] * d[12] * d[19] + std::pow(d[1], 2) * d[3] * d[12] * d[19] + d[1] * std::pow(d[2], 2) * d[13] * d[20] + d[1] * std::pow(d[3], 2) * d[13] * d[20] - std::pow(d[0], 2) * d[3] * d[14] * d[21] - std::pow(d[1], 2) * d[3] * d[14] * d[21] + d[0] * std::pow(d[2], 2) * d[15] + d[0] * std::pow(d[3], 2) * d[15];
					coeffs[5] = -2 * std::pow(d[0], 2) * d[14] * d[15] - 2 * std::pow(d[1], 2) * d[14] * d[15] - 2 * std::pow(d[2], 2) * d[14] * d[15] - 2 * std::pow(d[3], 2) * d[14] * d[15] + 2 * std::pow(d[0], 2) * d[9] * d[20] + 2 * std::pow(d[1], 2) * d[9] * d[20] + 2 * std::pow(d[2], 2) * d[9] * d[20] + 2 * std::pow(d[3], 2) * d[9] * d[20];
					coeffs[6] = -std::pow(d[0], 2) * std::pow(d[2], 2) * d[9] * d[15] - std::pow(d[1], 2) * std::pow(d[2], 2) * d[9] * d[15] - std::pow(d[0], 2) * std::pow(d[3], 2) * d[9] * d[15] - std::pow(d[1], 2) * std::pow(d[3], 2) * d[9] * d[15] - std::pow(d[0], 2) * std::pow(d[2], 2) * d[12] * d[18] - std::pow(d[1], 2) * std::pow(d[2], 2) * d[12] * d[18] - std::pow(d[0], 2) * std::pow(d[3], 2) * d[12] * d[18] - std::pow(d[1], 2) * std::pow(d[3], 2) * d[12] * d[18] - std::pow(d[0], 2) * std::pow(d[2], 2) * d[14] * d[20] - std::pow(d[1], 2) * std::pow(d[2], 2) * d[14] * d[20] - std::pow(d[0], 2) * std::pow(d[3], 2) * d[14] * d[20] - std::pow(d[1], 2) * std::pow(d[3], 2) * d[14] * d[20];
					coeffs[7] = d[1] * d[8] * d[15] - d[3] * d[9] * d[16] - d[0] * d[10] * d[18] - d[1] * d[11] * d[18] + d[3] * d[12] * d[19] + d[1] * d[13] * d[20] - d[3] * d[14] * d[21] + d[0] * d[15];
					coeffs[8] = -2 * d[14] * d[15] + 2 * d[9] * d[20];
					coeffs[9] = -2 * d[1] * std::pow(d[2], 2) * d[13] * d[15] - 2 * d[1] * std::pow(d[3], 2) * d[13] * d[15] + 2 * std::pow(d[0], 2) * d[3] * d[14] * d[16] + 2 * std::pow(d[1], 2) * d[3] * d[14] * d[16] + 2 * d[1] * std::pow(d[2], 2) * d[8] * d[20] + 2 * d[1] * std::pow(d[3], 2) * d[8] * d[20] - 2 * std::pow(d[0], 2) * d[3] * d[9] * d[21] - 2 * std::pow(d[1], 2) * d[3] * d[9] * d[21] + 2 * d[0] * std::pow(d[2], 2) * d[20] + 2 * d[0] * std::pow(d[3], 2) * d[20];
					coeffs[10] = -std::pow(d[0], 2) * d[9] * d[15] - std::pow(d[1], 2) * d[9] * d[15] - std::pow(d[2], 2) * d[9] * d[15] - std::pow(d[3], 2) * d[9] * d[15] - std::pow(d[0], 2) * d[12] * d[18] - std::pow(d[1], 2) * d[12] * d[18] - std::pow(d[2], 2) * d[12] * d[18] - std::pow(d[3], 2) * d[12] * d[18] - std::pow(d[0], 2) * d[14] * d[20] - std::pow(d[1], 2) * d[14] * d[20] - std::pow(d[2], 2) * d[14] * d[20] - std::pow(d[3], 2) * d[14] * d[20];
					coeffs[11] = -d[1] * d[3] * d[8] * d[16] + d[0] * d[3] * d[10] * d[19] + d[1] * d[3] * d[11] * d[19] - d[1] * d[3] * d[13] * d[21] - d[0] * d[3] * d[16];
					coeffs[12] = -2 * d[1] * d[13] * d[15] + 2 * d[3] * d[14] * d[16] + 2 * d[1] * d[8] * d[20] - 2 * d[3] * d[9] * d[21] + 2 * d[0] * d[20];
					coeffs[13] = -d[9] * d[15] - d[12] * d[18] - d[14] * d[20];
					coeffs[14] = -d[1] * std::pow(d[2], 2) * d[8] * d[15] - d[1] * std::pow(d[3], 2) * d[8] * d[15] + std::pow(d[0], 2) * d[3] * d[9] * d[16] + std::pow(d[1], 2) * d[3] * d[9] * d[16] - d[0] * std::pow(d[2], 2) * d[10] * d[18] - d[0] * std::pow(d[3], 2) * d[10] * d[18] - d[1] * std::pow(d[2], 2) * d[11] * d[18] - d[1] * std::pow(d[3], 2) * d[11] * d[18] + std::pow(d[0], 2) * d[3] * d[12] * d[19] + std::pow(d[1], 2) * d[3] * d[12] * d[19] - d[1] * std::pow(d[2], 2) * d[13] * d[20] - d[1] * std::pow(d[3], 2) * d[13] * d[20] + std::pow(d[0], 2) * d[3] * d[14] * d[21] + std::pow(d[1], 2) * d[3] * d[14] * d[21] - d[0] * std::pow(d[2], 2) * d[15] - d[0] * std::pow(d[3], 2) * d[15];
					coeffs[15] = 2 * d[1] * d[3] * d[13] * d[16] - 2 * d[1] * d[3] * d[8] * d[21] - 2 * d[0] * d[3] * d[21];
					coeffs[16] = -d[1] * d[8] * d[15] + d[3] * d[9] * d[16] - d[0] * d[10] * d[18] - d[1] * d[11] * d[18] + d[3] * d[12] * d[19] - d[1] * d[13] * d[20] + d[3] * d[14] * d[21] - d[0] * d[15];
					coeffs[17] = d[1] * d[3] * d[8] * d[16] + d[0] * d[3] * d[10] * d[19] + d[1] * d[3] * d[11] * d[19] + d[1] * d[3] * d[13] * d[21] + d[0] * d[3] * d[16];
					coeffs[18] = -std::pow(d[0], 2) * d[2] * d[9] * d[15] - std::pow(d[1], 2) * d[2] * d[9] * d[15] - std::pow(d[0], 2) * d[3] * d[12] * d[17] - std::pow(d[1], 2) * d[3] * d[12] * d[17] + std::pow(d[0], 2) * d[2] * d[12] * d[18] + std::pow(d[1], 2) * d[2] * d[12] * d[18] - std::pow(d[0], 2) * d[2] * d[14] * d[20] - std::pow(d[1], 2) * d[2] * d[14] * d[20] + std::pow(d[0], 2) * d[3] * d[9] + std::pow(d[1], 2) * d[3] * d[9];
					coeffs[19] = -d[2] * d[9] * d[15] - d[3] * d[12] * d[17] + d[2] * d[12] * d[18] - d[2] * d[14] * d[20] + d[3] * d[9];
					coeffs[20] = 2 * std::pow(d[0], 2) * d[2] * d[14] * d[15] + 2 * std::pow(d[1], 2) * d[2] * d[14] * d[15] - 2 * std::pow(d[0], 2) * d[2] * d[9] * d[20] - 2 * std::pow(d[1], 2) * d[2] * d[9] * d[20] - 2 * std::pow(d[0], 2) * d[3] * d[14] - 2 * std::pow(d[1], 2) * d[3] * d[14];
					coeffs[21] = -d[1] * d[2] * d[8] * d[15] - d[0] * d[3] * d[10] * d[17] - d[1] * d[3] * d[11] * d[17] + d[0] * d[2] * d[10] * d[18] + d[1] * d[2] * d[11] * d[18] - d[1] * d[2] * d[13] * d[20] + d[1] * d[3] * d[8] - d[0] * d[2] * d[15] + d[0] * d[3];
					coeffs[22] = 2 * d[2] * d[14] * d[15] - 2 * d[2] * d[9] * d[20] - 2 * d[3] * d[14];
					coeffs[23] = std::pow(d[0], 2) * d[2] * d[9] * d[15] + std::pow(d[1], 2) * d[2] * d[9] * d[15] - std::pow(d[0], 2) * d[3] * d[12] * d[17] - std::pow(d[1], 2) * d[3] * d[12] * d[17] + std::pow(d[0], 2) * d[2] * d[12] * d[18] + std::pow(d[1], 2) * d[2] * d[12] * d[18] + std::pow(d[0], 2) * d[2] * d[14] * d[20] + std::pow(d[1], 2) * d[2] * d[14] * d[20] - std::pow(d[0], 2) * d[3] * d[9] - std::pow(d[1], 2) * d[3] * d[9];
					coeffs[24] = 2 * d[1] * d[2] * d[13] * d[15] - 2 * d[1] * d[2] * d[8] * d[20] - 2 * d[1] * d[3] * d[13] - 2 * d[0] * d[2] * d[20];
					coeffs[25] = d[2] * d[9] * d[15] - d[3] * d[12] * d[17] + d[2] * d[12] * d[18] + d[2] * d[14] * d[20] - d[3] * d[9];
					coeffs[26] = d[1] * d[2] * d[8] * d[15] - d[0] * d[3] * d[10] * d[17] - d[1] * d[3] * d[11] * d[17] + d[0] * d[2] * d[10] * d[18] + d[1] * d[2] * d[11] * d[18] + d[1] * d[2] * d[13] * d[20] - d[1] * d[3] * d[8] + d[0] * d[2] * d[15] - d[0] * d[3];
					coeffs[27] = -std::pow(d[4], 2) * d[6] * d[9] * d[15] - std::pow(d[5], 2) * d[6] * d[9] * d[15] - std::pow(d[4], 2) * d[7] * d[12] * d[17] - std::pow(d[5], 2) * d[7] * d[12] * d[17] + std::pow(d[4], 2) * d[6] * d[12] * d[18] + std::pow(d[5], 2) * d[6] * d[12] * d[18] - std::pow(d[4], 2) * d[6] * d[14] * d[20] - std::pow(d[5], 2) * d[6] * d[14] * d[20] + std::pow(d[4], 2) * d[7] * d[9] + std::pow(d[5], 2) * d[7] * d[9];
					coeffs[28] = -d[6] * d[9] * d[15] - d[7] * d[12] * d[17] + d[6] * d[12] * d[18] - d[6] * d[14] * d[20] + d[7] * d[9];
					coeffs[29] = 2 * std::pow(d[4], 2) * d[6] * d[14] * d[15] + 2 * std::pow(d[5], 2) * d[6] * d[14] * d[15] - 2 * std::pow(d[4], 2) * d[6] * d[9] * d[20] - 2 * std::pow(d[5], 2) * d[6] * d[9] * d[20] - 2 * std::pow(d[4], 2) * d[7] * d[14] - 2 * std::pow(d[5], 2) * d[7] * d[14];
					coeffs[30] = -d[5] * d[6] * d[8] * d[15] - d[4] * d[7] * d[10] * d[17] - d[5] * d[7] * d[11] * d[17] + d[4] * d[6] * d[10] * d[18] + d[5] * d[6] * d[11] * d[18] - d[5] * d[6] * d[13] * d[20] + d[5] * d[7] * d[8] - d[4] * d[6] * d[15] + d[4] * d[7];
					coeffs[31] = 2 * d[6] * d[14] * d[15] - 2 * d[6] * d[9] * d[20] - 2 * d[7] * d[14];
					coeffs[32] = std::pow(d[4], 2) * d[6] * d[9] * d[15] + std::pow(d[5], 2) * d[6] * d[9] * d[15] - std::pow(d[4], 2) * d[7] * d[12] * d[17] - std::pow(d[5], 2) * d[7] * d[12] * d[17] + std::pow(d[4], 2) * d[6] * d[12] * d[18] + std::pow(d[5], 2) * d[6] * d[12] * d[18] + std::pow(d[4], 2) * d[6] * d[14] * d[20] + std::pow(d[5], 2) * d[6] * d[14] * d[20] - std::pow(d[4], 2) * d[7] * d[9] - std::pow(d[5], 2) * d[7] * d[9];
					coeffs[33] = 2 * d[5] * d[6] * d[13] * d[15] - 2 * d[5] * d[6] * d[8] * d[20] - 2 * d[5] * d[7] * d[13] - 2 * d[4] * d[6] * d[20];
					coeffs[34] = d[6] * d[9] * d[15] - d[7] * d[12] * d[17] + d[6] * d[12] * d[18] + d[6] * d[14] * d[20] - d[7] * d[9];
					coeffs[35] = d[5] * d[6] * d[8] * d[15] - d[4] * d[7] * d[10] * d[17] - d[5] * d[7] * d[11] * d[17] + d[4] * d[6] * d[10] * d[18] + d[5] * d[6] * d[11] * d[18] + d[5] * d[6] * d[13] * d[20] - d[5] * d[7] * d[8] + d[4] * d[6] * d[15] - d[4] * d[7];

					static const int coeffs_ind[] = {0, 18, 27, 1, 0, 19, 18, 27, 28, 2, 0, 18, 20, 27, 29, 3, 1, 18, 27, 19, 28, 4, 18, 21, 0, 27, 30, 5, 2, 1, 18, 19, 0, 22, 20, 27, 29, 28, 31, 6, 2, 20, 23, 29, 32, 3, 19, 28, 7, 4,
													 19, 18, 21, 1, 27, 30, 28, 8, 5, 3, 20, 19, 29, 18, 1, 22, 27, 28, 31, 9, 4, 20, 21, 0, 24, 2, 18, 27, 29, 30, 33, 10, 6, 5, 20, 22, 2, 25, 23, 29, 32, 31, 34, 6, 23, 32, 7, 21, 30, 19,
													 3, 28, 8, 22, 31, 19, 3, 28, 11, 21, 4, 18, 27, 30, 12, 9, 7, 22, 21, 20, 1, 4, 24, 5, 19, 30, 27, 29, 28, 33, 31, 18, 13, 10, 8, 23, 22, 32, 20, 5, 25, 29, 31, 34, 14, 9, 23, 24, 2, 26,
													 6, 20, 29, 32, 33, 35, 10, 23, 25, 6, 32, 34, 11, 21, 7, 30, 19, 28, 12, 24, 33, 22, 21, 3, 7, 30, 8, 28, 31, 19, 15, 11, 24, 4, 9, 21, 20, 30, 18, 29, 33, 27, 13, 25, 23, 10, 32, 34, 14, 26,
													 6, 23, 32, 35, 16, 14, 12, 25, 24, 23, 5, 9, 26, 10, 22, 33, 29, 32, 31, 35, 34, 20, 11, 21, 30, 15, 24, 7, 11, 12, 30, 33, 22, 19, 31, 21, 28, 16, 26, 35, 25, 24, 8, 12, 33, 13, 31, 34, 22, 17,
													 15, 26, 9, 14, 24, 23, 33, 20, 32, 35, 29, 16, 26, 10, 14, 25, 35, 32, 34, 23, 11, 15, 24, 21, 33, 30, 17, 26, 12, 15, 16, 33, 35, 25, 22, 34, 24, 31, 26, 13, 16, 35, 34, 25, 17, 14, 26, 35, 23, 32,
													 15, 17, 26, 24, 35, 33, 16, 17, 35, 25, 26, 34, 17, 26, 35};

					static const int C_ind[] = {0, 12, 29, 30, 31, 42, 43, 53, 59, 60, 62, 66, 72, 88, 89, 90, 91, 93, 97, 103, 113, 120, 124, 132, 135, 145, 149, 150, 151, 152, 155, 156, 161, 162, 163, 167, 173, 178, 179, 180, 182, 186, 192, 208, 209, 211, 213, 217, 240, 241,
												244, 248, 253, 255, 259, 263, 265, 270, 271, 272, 273, 275, 277, 279, 281, 283, 284, 287, 293, 300, 302, 304, 306, 310, 312, 315, 316, 321, 325, 328, 329, 330, 331, 332, 335, 336, 341, 342, 343, 347, 353, 358, 359, 362, 366, 388, 391, 393, 397, 398,
												405, 409, 421, 423, 427, 429, 431, 434, 450, 454, 465, 470, 474, 475, 480, 481, 482, 484, 485, 488, 490, 491, 493, 495, 496, 497, 498, 499, 501, 503, 505, 506, 510, 511, 512, 513, 515, 517, 519, 521, 523, 524, 527, 533, 540, 542, 544, 546, 550, 552,
												555, 556, 561, 565, 568, 569, 572, 575, 576, 581, 587, 598, 601, 608, 615, 619, 620, 624, 631, 633, 637, 638, 639, 640, 641, 644, 645, 648, 649, 656, 660, 662, 664, 670, 675, 676, 680, 681, 682, 684, 685, 687, 692, 695, 699, 701, 704, 707, 722, 726,
												730, 736, 741, 748, 750, 751, 752, 754, 755, 758, 760, 761, 763, 765, 766, 767, 768, 769, 771, 773, 775, 776, 795, 800, 804, 811, 818, 820, 821, 825, 828, 829, 830, 832, 834, 836, 837, 841, 843, 847, 848, 849, 850, 851, 854, 855, 858, 859, 866, 870,
												872, 874, 880, 885, 886, 890, 891, 892, 894, 895, 897, 902, 905, 910, 911, 916, 917, 918, 921, 926, 940, 945, 950, 952, 954, 957, 961, 968, 970, 971, 975, 978, 979, 980, 982, 984, 986, 987, 999, 1000, 1001, 1004, 1008, 1016, 1022, 1030, 1036, 1041, 1042, 1047,
												1060, 1065, 1070, 1072, 1074, 1077, 1090, 1091, 1098, 1102, 1106, 1107, 1120, 1132, 1137};

					MatrixXd C = MatrixXd::Zero(30, 38);
					for (int i = 0; i < 315; i++)
					{
						C(C_ind[i]) = coeffs(coeffs_ind[i]);
					}

					MatrixXd C0 = C.leftCols(30);
					MatrixXd C1 = C.rightCols(8);
					MatrixXd C12 = C0.fullPivLu().solve(C1);
					MatrixXd RR(13, 8);
					RR << -C12.bottomRows(5), MatrixXd::Identity(8, 8);

					static const int AM_ind[] = {0, 1, 2, 3, 4, 6, 7, 10};
					MatrixXd AM(8, 8);
					for (int i = 0; i < 8; i++)
					{
						AM.row(i) = RR.row(AM_ind[i]);
					}

					EigenSolver<MatrixXd> es(AM);
					ArrayXcd D = es.eigenvalues();
					ArrayXXcd V = es.eigenvectors();
					V = (V / V.row(7).replicate(8, 1)).eval();

					MatrixXcd sols(3, 8);
					sols.row(0) = D.transpose();												  // s
					sols.row(1) = V.row(6);														  // f
					sols.row(2) = V.row(0).array() / (sols.row(1).array() * sols.row(1).array()); // d

					for (int k = 0; k < 8; ++k)
					{
						if (sols(0, k).imag() > std::numeric_limits<double>::epsilon() || sols(0, k).real() > 1 || sols(0, k).real() < -1 || sols(1, k).real() < 0) // cos(theta) 1-s^2 > 0
							continue;

						double r = sols(0, k).real();
						double rsqr = r * r;
						double focal = sols(1, k).real();
						if (focal < 0)
							continue;

						double dis = sols(2, k).real();

						Eigen::Matrix<double, 3, 3> Ry;
						Ry << (1 - rsqr), 0, 2 * r,
							0, 1 + rsqr, 0,
							-2 * r, 0, (1 - rsqr);

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
			}

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
 // namespace gcransac
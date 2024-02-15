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
			class StitchingGeneralRadialEqual3pt : public SolverEngine
			{
			public:
				StitchingGeneralRadialEqual3pt() : gravity_source(Eigen::Matrix3d::Identity()),
												   gravity_destination(Eigen::Matrix3d::Identity())
				{
				}

				~StitchingGeneralRadialEqual3pt()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				static constexpr size_t maximumSolutions()
				{
					return 18;
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
					return "General Equal Radial (3PC)";
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

			void StitchingGeneralRadialEqual3pt::solve_quadratic(double a, double b, double c, std::complex<double> roots[2]) const
			{

				std::complex<double> b2m4ac = b * b - 4 * a * c;
				std::complex<double> sq = std::sqrt(b2m4ac);

				// Choose sign to avoid cancellations
				roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
				roots[1] = c / (a * roots[0]);
			}

			OLGA_INLINE bool StitchingGeneralRadialEqual3pt::estimateModel(
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

				if (gravity_source(0, 0) != 1)
				{
					// TD DO!
				}

				else
				{
					double r1 = u1 * u1 + v1 * v1,
						   r2 = u2 * u2 + v2 * v2,
						   r3 = u3 * u3 + v3 * v3,
						   r4 = u4 * u4 + v4 * v4,
						   r5 = u5 * u5 + v5 * v5,
						   r6 = u6 * u6 + v6 * v6;

					double d[18] = {0};

					d[0] = u1;
					d[1] = v1;
					d[2] = u2;
					d[3] = v2;
					d[4] = u3;
					d[5] = v3;
					d[6] = u4;
					d[7] = v4;
					d[8] = u5;
					d[9] = v5;
					d[10] = u6;
					d[11] = v6;
					d[12] = r1;
					d[13] = r2;
					d[14] = r3;
					d[15] = r4;
					d[16] = r5;
					d[17] = r6;

					VectorXd coeffs(32);
					coeffs[0] = -2 * d[2] * d[6] * std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) * d[15] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) * d[15] + 2 * d[0] * d[4] * d[12] * std::pow(d[13], 2) * d[14] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[12] * std::pow(d[13], 2) * d[14] * std::pow(d[15], 2) + std::pow(d[12], 2) * std::pow(d[13], 2) * std::pow(d[14], 2) * d[15] - std::pow(d[12], 2) * std::pow(d[13], 2) * d[14] * std::pow(d[15], 2) + std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) * std::pow(d[15], 2) - d[12] * std::pow(d[13], 2) * std::pow(d[14], 2) * std::pow(d[15], 2);
					coeffs[1] = -2 * d[2] * d[6] * std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) - 4 * d[2] * d[6] * std::pow(d[12], 2) * d[13] * d[14] * d[15] - 4 * d[3] * d[7] * std::pow(d[12], 2) * d[13] * d[14] * d[15] + 4 * d[0] * d[4] * d[12] * std::pow(d[13], 2) * d[14] * d[15] + 4 * d[1] * d[5] * d[12] * std::pow(d[13], 2) * d[14] * d[15] - 2 * d[2] * d[6] * std::pow(d[12], 2) * std::pow(d[14], 2) * d[15] - 2 * d[3] * d[7] * std::pow(d[12], 2) * std::pow(d[14], 2) * d[15] - 4 * d[2] * d[6] * d[12] * d[13] * std::pow(d[14], 2) * d[15] - 4 * d[3] * d[7] * d[12] * d[13] * std::pow(d[14], 2) * d[15] + 2 * d[0] * d[4] * d[12] * std::pow(d[13], 2) * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[12] * std::pow(d[13], 2) * std::pow(d[15], 2) + 4 * d[0] * d[4] * d[12] * d[13] * d[14] * std::pow(d[15], 2) + 4 * d[1] * d[5] * d[12] * d[13] * d[14] * std::pow(d[15], 2) + 2 * d[0] * d[4] * std::pow(d[13], 2) * d[14] * std::pow(d[15], 2) + 2 * d[1] * d[5] * std::pow(d[13], 2) * d[14] * std::pow(d[15], 2) + 4 * std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) * d[15] - 4 * d[12] * std::pow(d[13], 2) * d[14] * std::pow(d[15], 2);
					coeffs[2] = -4 * d[2] * d[6] * std::pow(d[12], 2) * d[13] * d[14] - 4 * d[3] * d[7] * std::pow(d[12], 2) * d[13] * d[14] + 2 * d[0] * d[4] * d[12] * std::pow(d[13], 2) * d[14] + 2 * d[1] * d[5] * d[12] * std::pow(d[13], 2) * d[14] - 2 * d[2] * d[6] * std::pow(d[12], 2) * std::pow(d[14], 2) - 2 * d[3] * d[7] * std::pow(d[12], 2) * std::pow(d[14], 2) - 4 * d[2] * d[6] * d[12] * d[13] * std::pow(d[14], 2) - 4 * d[3] * d[7] * d[12] * d[13] * std::pow(d[14], 2) - 2 * d[2] * d[6] * std::pow(d[12], 2) * d[13] * d[15] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[13] * d[15] + 4 * d[0] * d[4] * d[12] * std::pow(d[13], 2) * d[15] + 4 * d[1] * d[5] * d[12] * std::pow(d[13], 2) * d[15] - 4 * d[2] * d[6] * std::pow(d[12], 2) * d[14] * d[15] - 4 * d[3] * d[7] * std::pow(d[12], 2) * d[14] * d[15] + 8 * d[0] * d[4] * d[12] * d[13] * d[14] * d[15] + 8 * d[1] * d[5] * d[12] * d[13] * d[14] * d[15] - 8 * d[2] * d[6] * d[12] * d[13] * d[14] * d[15] - 8 * d[3] * d[7] * d[12] * d[13] * d[14] * d[15] + 4 * d[0] * d[4] * std::pow(d[13], 2) * d[14] * d[15] + 4 * d[1] * d[5] * std::pow(d[13], 2) * d[14] * d[15] - 4 * d[2] * d[6] * d[12] * std::pow(d[14], 2) * d[15] - 4 * d[3] * d[7] * d[12] * std::pow(d[14], 2) * d[15] - 2 * d[2] * d[6] * d[13] * std::pow(d[14], 2) * d[15] - 2 * d[3] * d[7] * d[13] * std::pow(d[14], 2) * d[15] + 4 * d[0] * d[4] * d[12] * d[13] * std::pow(d[15], 2) + 4 * d[1] * d[5] * d[12] * d[13] * std::pow(d[15], 2) + 2 * d[0] * d[4] * std::pow(d[13], 2) * std::pow(d[15], 2) + 2 * d[1] * d[5] * std::pow(d[13], 2) * std::pow(d[15], 2) + 2 * d[0] * d[4] * d[12] * d[14] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[12] * d[14] * std::pow(d[15], 2) + 4 * d[0] * d[4] * d[13] * d[14] * std::pow(d[15], 2) + 4 * d[1] * d[5] * d[13] * d[14] * std::pow(d[15], 2) - std::pow(d[12], 2) * std::pow(d[13], 2) * d[14] + std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) - d[12] * std::pow(d[13], 2) * std::pow(d[14], 2) + std::pow(d[12], 2) * std::pow(d[13], 2) * d[15] + 4 * std::pow(d[12], 2) * d[13] * d[14] * d[15] - 4 * d[12] * std::pow(d[13], 2) * d[14] * d[15] + std::pow(d[12], 2) * std::pow(d[14], 2) * d[15] + 4 * d[12] * d[13] * std::pow(d[14], 2) * d[15] + std::pow(d[13], 2) * std::pow(d[14], 2) * d[15] + std::pow(d[12], 2) * d[13] * std::pow(d[15], 2) - d[12] * std::pow(d[13], 2) * std::pow(d[15], 2) - std::pow(d[12], 2) * d[14] * std::pow(d[15], 2) - 4 * d[12] * d[13] * d[14] * std::pow(d[15], 2) - std::pow(d[13], 2) * d[14] * std::pow(d[15], 2) - d[12] * std::pow(d[14], 2) * std::pow(d[15], 2) + d[13] * std::pow(d[14], 2) * std::pow(d[15], 2);
					coeffs[3] = -2 * d[2] * d[6] * std::pow(d[12], 2) * d[13] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[13] + 2 * d[0] * d[4] * d[12] * std::pow(d[13], 2) + 2 * d[1] * d[5] * d[12] * std::pow(d[13], 2) - 4 * d[2] * d[6] * std::pow(d[12], 2) * d[14] - 4 * d[3] * d[7] * std::pow(d[12], 2) * d[14] + 4 * d[0] * d[4] * d[12] * d[13] * d[14] + 4 * d[1] * d[5] * d[12] * d[13] * d[14] - 8 * d[2] * d[6] * d[12] * d[13] * d[14] - 8 * d[3] * d[7] * d[12] * d[13] * d[14] + 2 * d[0] * d[4] * std::pow(d[13], 2) * d[14] + 2 * d[1] * d[5] * std::pow(d[13], 2) * d[14] - 4 * d[2] * d[6] * d[12] * std::pow(d[14], 2) - 4 * d[3] * d[7] * d[12] * std::pow(d[14], 2) - 2 * d[2] * d[6] * d[13] * std::pow(d[14], 2) - 2 * d[3] * d[7] * d[13] * std::pow(d[14], 2) - 2 * d[2] * d[6] * std::pow(d[12], 2) * d[15] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[15] + 8 * d[0] * d[4] * d[12] * d[13] * d[15] + 8 * d[1] * d[5] * d[12] * d[13] * d[15] - 4 * d[2] * d[6] * d[12] * d[13] * d[15] - 4 * d[3] * d[7] * d[12] * d[13] * d[15] + 4 * d[0] * d[4] * std::pow(d[13], 2) * d[15] + 4 * d[1] * d[5] * std::pow(d[13], 2) * d[15] + 4 * d[0] * d[4] * d[12] * d[14] * d[15] + 4 * d[1] * d[5] * d[12] * d[14] * d[15] - 8 * d[2] * d[6] * d[12] * d[14] * d[15] - 8 * d[3] * d[7] * d[12] * d[14] * d[15] + 8 * d[0] * d[4] * d[13] * d[14] * d[15] + 8 * d[1] * d[5] * d[13] * d[14] * d[15] - 4 * d[2] * d[6] * d[13] * d[14] * d[15] - 4 * d[3] * d[7] * d[13] * d[14] * d[15] - 2 * d[2] * d[6] * std::pow(d[14], 2) * d[15] - 2 * d[3] * d[7] * std::pow(d[14], 2) * d[15] + 2 * d[0] * d[4] * d[12] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[12] * std::pow(d[15], 2) + 4 * d[0] * d[4] * d[13] * std::pow(d[15], 2) + 4 * d[1] * d[5] * d[13] * std::pow(d[15], 2) + 2 * d[0] * d[4] * d[14] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[14] * std::pow(d[15], 2) - 4 * d[12] * std::pow(d[13], 2) * d[14] + 4 * std::pow(d[12], 2) * d[13] * d[15] + 4 * d[13] * std::pow(d[14], 2) * d[15] - 4 * d[12] * d[14] * std::pow(d[15], 2);
					coeffs[4] = -std::pow(d[2], 2) * std::pow(d[6], 2) * std::pow(d[12], 2) * std::pow(d[14], 2) - 2 * d[2] * d[3] * d[6] * d[7] * std::pow(d[12], 2) * std::pow(d[14], 2) - std::pow(d[3], 2) * std::pow(d[7], 2) * std::pow(d[12], 2) * std::pow(d[14], 2) + std::pow(d[0], 2) * std::pow(d[4], 2) * std::pow(d[13], 2) * std::pow(d[15], 2) + 2 * d[0] * d[1] * d[4] * d[5] * std::pow(d[13], 2) * std::pow(d[15], 2) + std::pow(d[1], 2) * std::pow(d[5], 2) * std::pow(d[13], 2) * std::pow(d[15], 2) - 2 * d[2] * d[6] * std::pow(d[12], 2) * d[13] * d[14] * d[15] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[13] * d[14] * d[15] + 2 * d[0] * d[4] * d[12] * std::pow(d[13], 2) * d[14] * d[15] + 2 * d[1] * d[5] * d[12] * std::pow(d[13], 2) * d[14] * d[15] - 2 * d[2] * d[6] * d[12] * d[13] * std::pow(d[14], 2) * d[15] - 2 * d[3] * d[7] * d[12] * d[13] * std::pow(d[14], 2) * d[15] + 2 * d[0] * d[4] * d[12] * d[13] * d[14] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[12] * d[13] * d[14] * std::pow(d[15], 2) + std::pow(d[12], 2) * d[13] * std::pow(d[14], 2) * d[15] - d[12] * std::pow(d[13], 2) * d[14] * std::pow(d[15], 2);
					coeffs[5] = -2 * d[2] * d[6] * std::pow(d[12], 2) - 2 * d[3] * d[7] * std::pow(d[12], 2) + 4 * d[0] * d[4] * d[12] * d[13] + 4 * d[1] * d[5] * d[12] * d[13] - 4 * d[2] * d[6] * d[12] * d[13] - 4 * d[3] * d[7] * d[12] * d[13] + 2 * d[0] * d[4] * std::pow(d[13], 2) + 2 * d[1] * d[5] * std::pow(d[13], 2) + 2 * d[0] * d[4] * d[12] * d[14] + 2 * d[1] * d[5] * d[12] * d[14] - 8 * d[2] * d[6] * d[12] * d[14] - 8 * d[3] * d[7] * d[12] * d[14] + 4 * d[0] * d[4] * d[13] * d[14] + 4 * d[1] * d[5] * d[13] * d[14] - 4 * d[2] * d[6] * d[13] * d[14] - 4 * d[3] * d[7] * d[13] * d[14] - 2 * d[2] * d[6] * std::pow(d[14], 2) - 2 * d[3] * d[7] * std::pow(d[14], 2) + 4 * d[0] * d[4] * d[12] * d[15] + 4 * d[1] * d[5] * d[12] * d[15] - 4 * d[2] * d[6] * d[12] * d[15] - 4 * d[3] * d[7] * d[12] * d[15] + 8 * d[0] * d[4] * d[13] * d[15] + 8 * d[1] * d[5] * d[13] * d[15] - 2 * d[2] * d[6] * d[13] * d[15] - 2 * d[3] * d[7] * d[13] * d[15] + 4 * d[0] * d[4] * d[14] * d[15] + 4 * d[1] * d[5] * d[14] * d[15] - 4 * d[2] * d[6] * d[14] * d[15] - 4 * d[3] * d[7] * d[14] * d[15] + 2 * d[0] * d[4] * std::pow(d[15], 2) + 2 * d[1] * d[5] * std::pow(d[15], 2) + std::pow(d[12], 2) * d[13] - d[12] * std::pow(d[13], 2) - std::pow(d[12], 2) * d[14] - 4 * d[12] * d[13] * d[14] - std::pow(d[13], 2) * d[14] - d[12] * std::pow(d[14], 2) + d[13] * std::pow(d[14], 2) + std::pow(d[12], 2) * d[15] + 4 * d[12] * d[13] * d[15] + std::pow(d[13], 2) * d[15] - 4 * d[12] * d[14] * d[15] + 4 * d[13] * d[14] * d[15] + std::pow(d[14], 2) * d[15] - d[12] * std::pow(d[15], 2) + d[13] * std::pow(d[15], 2) - d[14] * std::pow(d[15], 2);
					coeffs[6] = -2 * std::pow(d[2], 2) * std::pow(d[6], 2) * std::pow(d[12], 2) * d[14] - 4 * d[2] * d[3] * d[6] * d[7] * std::pow(d[12], 2) * d[14] - 2 * std::pow(d[3], 2) * std::pow(d[7], 2) * std::pow(d[12], 2) * d[14] - 2 * std::pow(d[2], 2) * std::pow(d[6], 2) * d[12] * std::pow(d[14], 2) - 4 * d[2] * d[3] * d[6] * d[7] * d[12] * std::pow(d[14], 2) - 2 * std::pow(d[3], 2) * std::pow(d[7], 2) * d[12] * std::pow(d[14], 2) + 2 * std::pow(d[0], 2) * std::pow(d[4], 2) * std::pow(d[13], 2) * d[15] + 4 * d[0] * d[1] * d[4] * d[5] * std::pow(d[13], 2) * d[15] + 2 * std::pow(d[1], 2) * std::pow(d[5], 2) * std::pow(d[13], 2) * d[15] + 2 * std::pow(d[0], 2) * std::pow(d[4], 2) * d[13] * std::pow(d[15], 2) + 4 * d[0] * d[1] * d[4] * d[5] * d[13] * std::pow(d[15], 2) + 2 * std::pow(d[1], 2) * std::pow(d[5], 2) * d[13] * std::pow(d[15], 2) - 2 * d[2] * d[6] * std::pow(d[12], 2) * d[13] * d[14] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[13] * d[14] - 2 * d[2] * d[6] * d[12] * d[13] * std::pow(d[14], 2) - 2 * d[3] * d[7] * d[12] * d[13] * std::pow(d[14], 2) + 2 * d[0] * d[4] * d[12] * std::pow(d[13], 2) * d[15] + 2 * d[1] * d[5] * d[12] * std::pow(d[13], 2) * d[15] - 2 * d[2] * d[6] * std::pow(d[12], 2) * d[14] * d[15] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[14] * d[15] + 8 * d[0] * d[4] * d[12] * d[13] * d[14] * d[15] + 8 * d[1] * d[5] * d[12] * d[13] * d[14] * d[15] - 8 * d[2] * d[6] * d[12] * d[13] * d[14] * d[15] - 8 * d[3] * d[7] * d[12] * d[13] * d[14] * d[15] + 2 * d[0] * d[4] * std::pow(d[13], 2) * d[14] * d[15] + 2 * d[1] * d[5] * std::pow(d[13], 2) * d[14] * d[15] - 2 * d[2] * d[6] * d[12] * std::pow(d[14], 2) * d[15] - 2 * d[3] * d[7] * d[12] * std::pow(d[14], 2) * d[15] + 2 * d[0] * d[4] * d[12] * d[13] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[12] * d[13] * std::pow(d[15], 2) + 2 * d[0] * d[4] * d[13] * d[14] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[13] * d[14] * std::pow(d[15], 2) + 2 * std::pow(d[12], 2) * d[13] * d[14] * d[15] - 2 * d[12] * std::pow(d[13], 2) * d[14] * d[15] + 2 * d[12] * d[13] * std::pow(d[14], 2) * d[15] - 2 * d[12] * d[13] * d[14] * std::pow(d[15], 2);
					coeffs[7] = 2 * d[0] * d[4] * d[12] + 2 * d[1] * d[5] * d[12] - 4 * d[2] * d[6] * d[12] - 4 * d[3] * d[7] * d[12] + 4 * d[0] * d[4] * d[13] + 4 * d[1] * d[5] * d[13] - 2 * d[2] * d[6] * d[13] - 2 * d[3] * d[7] * d[13] + 2 * d[0] * d[4] * d[14] + 2 * d[1] * d[5] * d[14] - 4 * d[2] * d[6] * d[14] - 4 * d[3] * d[7] * d[14] + 4 * d[0] * d[4] * d[15] + 4 * d[1] * d[5] * d[15] - 2 * d[2] * d[6] * d[15] - 2 * d[3] * d[7] * d[15] - 4 * d[12] * d[14] + 4 * d[13] * d[15];
					coeffs[8] = -std::pow(d[2], 2) * std::pow(d[6], 2) * std::pow(d[12], 2) - 2 * d[2] * d[3] * d[6] * d[7] * std::pow(d[12], 2) - std::pow(d[3], 2) * std::pow(d[7], 2) * std::pow(d[12], 2) + std::pow(d[0], 2) * std::pow(d[4], 2) * std::pow(d[13], 2) + 2 * d[0] * d[1] * d[4] * d[5] * std::pow(d[13], 2) + std::pow(d[1], 2) * std::pow(d[5], 2) * std::pow(d[13], 2) - 4 * std::pow(d[2], 2) * std::pow(d[6], 2) * d[12] * d[14] - 8 * d[2] * d[3] * d[6] * d[7] * d[12] * d[14] - 4 * std::pow(d[3], 2) * std::pow(d[7], 2) * d[12] * d[14] - std::pow(d[2], 2) * std::pow(d[6], 2) * std::pow(d[14], 2) - 2 * d[2] * d[3] * d[6] * d[7] * std::pow(d[14], 2) - std::pow(d[3], 2) * std::pow(d[7], 2) * std::pow(d[14], 2) + 4 * std::pow(d[0], 2) * std::pow(d[4], 2) * d[13] * d[15] + 8 * d[0] * d[1] * d[4] * d[5] * d[13] * d[15] + 4 * std::pow(d[1], 2) * std::pow(d[5], 2) * d[13] * d[15] + std::pow(d[0], 2) * std::pow(d[4], 2) * std::pow(d[15], 2) + 2 * d[0] * d[1] * d[4] * d[5] * std::pow(d[15], 2) + std::pow(d[1], 2) * std::pow(d[5], 2) * std::pow(d[15], 2) - 2 * d[2] * d[6] * std::pow(d[12], 2) * d[14] - 2 * d[3] * d[7] * std::pow(d[12], 2) * d[14] + 2 * d[0] * d[4] * d[12] * d[13] * d[14] + 2 * d[1] * d[5] * d[12] * d[13] * d[14] - 8 * d[2] * d[6] * d[12] * d[13] * d[14] - 8 * d[3] * d[7] * d[12] * d[13] * d[14] - 2 * d[2] * d[6] * d[12] * std::pow(d[14], 2) - 2 * d[3] * d[7] * d[12] * std::pow(d[14], 2) + 8 * d[0] * d[4] * d[12] * d[13] * d[15] + 8 * d[1] * d[5] * d[12] * d[13] * d[15] - 2 * d[2] * d[6] * d[12] * d[13] * d[15] - 2 * d[3] * d[7] * d[12] * d[13] * d[15] + 2 * d[0] * d[4] * std::pow(d[13], 2) * d[15] + 2 * d[1] * d[5] * std::pow(d[13], 2) * d[15] + 2 * d[0] * d[4] * d[12] * d[14] * d[15] + 2 * d[1] * d[5] * d[12] * d[14] * d[15] - 8 * d[2] * d[6] * d[12] * d[14] * d[15] - 8 * d[3] * d[7] * d[12] * d[14] * d[15] + 8 * d[0] * d[4] * d[13] * d[14] * d[15] + 8 * d[1] * d[5] * d[13] * d[14] * d[15] - 2 * d[2] * d[6] * d[13] * d[14] * d[15] - 2 * d[3] * d[7] * d[13] * d[14] * d[15] + 2 * d[0] * d[4] * d[13] * std::pow(d[15], 2) + 2 * d[1] * d[5] * d[13] * std::pow(d[15], 2) - d[12] * std::pow(d[13], 2) * d[14] + std::pow(d[12], 2) * d[13] * d[15] + d[13] * std::pow(d[14], 2) * d[15] - d[12] * d[14] * std::pow(d[15], 2);
					coeffs[9] = 2 * d[0] * d[4] + 2 * d[1] * d[5] - 2 * d[2] * d[6] - 2 * d[3] * d[7] - d[12] + d[13] - d[14] + d[15];
					coeffs[10] = -2 * std::pow(d[2], 2) * std::pow(d[6], 2) * d[12] - 4 * d[2] * d[3] * d[6] * d[7] * d[12] - 2 * std::pow(d[3], 2) * std::pow(d[7], 2) * d[12] + 2 * std::pow(d[0], 2) * std::pow(d[4], 2) * d[13] + 4 * d[0] * d[1] * d[4] * d[5] * d[13] + 2 * std::pow(d[1], 2) * std::pow(d[5], 2) * d[13] - 2 * std::pow(d[2], 2) * std::pow(d[6], 2) * d[14] - 4 * d[2] * d[3] * d[6] * d[7] * d[14] - 2 * std::pow(d[3], 2) * std::pow(d[7], 2) * d[14] + 2 * std::pow(d[0], 2) * std::pow(d[4], 2) * d[15] + 4 * d[0] * d[1] * d[4] * d[5] * d[15] + 2 * std::pow(d[1], 2) * std::pow(d[5], 2) * d[15] + 2 * d[0] * d[4] * d[12] * d[13] + 2 * d[1] * d[5] * d[12] * d[13] - 2 * d[2] * d[6] * d[12] * d[13] - 2 * d[3] * d[7] * d[12] * d[13] - 8 * d[2] * d[6] * d[12] * d[14] - 8 * d[3] * d[7] * d[12] * d[14] + 2 * d[0] * d[4] * d[13] * d[14] + 2 * d[1] * d[5] * d[13] * d[14] - 2 * d[2] * d[6] * d[13] * d[14] - 2 * d[3] * d[7] * d[13] * d[14] + 2 * d[0] * d[4] * d[12] * d[15] + 2 * d[1] * d[5] * d[12] * d[15] - 2 * d[2] * d[6] * d[12] * d[15] - 2 * d[3] * d[7] * d[12] * d[15] + 8 * d[0] * d[4] * d[13] * d[15] + 8 * d[1] * d[5] * d[13] * d[15] + 2 * d[0] * d[4] * d[14] * d[15] + 2 * d[1] * d[5] * d[14] * d[15] - 2 * d[2] * d[6] * d[14] * d[15] - 2 * d[3] * d[7] * d[14] * d[15] - 2 * d[12] * d[13] * d[14] + 2 * d[12] * d[13] * d[15] - 2 * d[12] * d[14] * d[15] + 2 * d[13] * d[14] * d[15];
					coeffs[11] = -std::pow(d[2], 2) * std::pow(d[6], 2) * std::pow(d[12], 2) * d[14] - 2 * d[2] * d[3] * d[6] * d[7] * std::pow(d[12], 2) * d[14] - std::pow(d[3], 2) * std::pow(d[7], 2) * std::pow(d[12], 2) * d[14] - std::pow(d[2], 2) * std::pow(d[6], 2) * d[12] * std::pow(d[14], 2) - 2 * d[2] * d[3] * d[6] * d[7] * d[12] * std::pow(d[14], 2) - std::pow(d[3], 2) * std::pow(d[7], 2) * d[12] * std::pow(d[14], 2) + std::pow(d[0], 2) * std::pow(d[4], 2) * std::pow(d[13], 2) * d[15] + 2 * d[0] * d[1] * d[4] * d[5] * std::pow(d[13], 2) * d[15] + std::pow(d[1], 2) * std::pow(d[5], 2) * std::pow(d[13], 2) * d[15] + std::pow(d[0], 2) * std::pow(d[4], 2) * d[13] * std::pow(d[15], 2) + 2 * d[0] * d[1] * d[4] * d[5] * d[13] * std::pow(d[15], 2) + std::pow(d[1], 2) * std::pow(d[5], 2) * d[13] * std::pow(d[15], 2) + 2 * d[0] * d[4] * d[12] * d[13] * d[14] * d[15] + 2 * d[1] * d[5] * d[12] * d[13] * d[14] * d[15] - 2 * d[2] * d[6] * d[12] * d[13] * d[14] * d[15] - 2 * d[3] * d[7] * d[12] * d[13] * d[14] * d[15];
					coeffs[12] = std::pow(d[0], 2) * std::pow(d[4], 2) + 2 * d[0] * d[1] * d[4] * d[5] + std::pow(d[1], 2) * std::pow(d[5], 2) - std::pow(d[2], 2) * std::pow(d[6], 2) - 2 * d[2] * d[3] * d[6] * d[7] - std::pow(d[3], 2) * std::pow(d[7], 2) - 2 * d[2] * d[6] * d[12] - 2 * d[3] * d[7] * d[12] + 2 * d[0] * d[4] * d[13] + 2 * d[1] * d[5] * d[13] - 2 * d[2] * d[6] * d[14] - 2 * d[3] * d[7] * d[14] + 2 * d[0] * d[4] * d[15] + 2 * d[1] * d[5] * d[15] - d[12] * d[14] + d[13] * d[15];
					coeffs[13] = -4 * std::pow(d[2], 2) * std::pow(d[6], 2) * d[12] * d[14] - 8 * d[2] * d[3] * d[6] * d[7] * d[12] * d[14] - 4 * std::pow(d[3], 2) * std::pow(d[7], 2) * d[12] * d[14] + 4 * std::pow(d[0], 2) * std::pow(d[4], 2) * d[13] * d[15] + 8 * d[0] * d[1] * d[4] * d[5] * d[13] * d[15] + 4 * std::pow(d[1], 2) * std::pow(d[5], 2) * d[13] * d[15] - 2 * d[2] * d[6] * d[12] * d[13] * d[14] - 2 * d[3] * d[7] * d[12] * d[13] * d[14] + 2 * d[0] * d[4] * d[12] * d[13] * d[15] + 2 * d[1] * d[5] * d[12] * d[13] * d[15] - 2 * d[2] * d[6] * d[12] * d[14] * d[15] - 2 * d[3] * d[7] * d[12] * d[14] * d[15] + 2 * d[0] * d[4] * d[13] * d[14] * d[15] + 2 * d[1] * d[5] * d[13] * d[14] * d[15];
					coeffs[14] = -std::pow(d[2], 2) * std::pow(d[6], 2) * d[12] - 2 * d[2] * d[3] * d[6] * d[7] * d[12] - std::pow(d[3], 2) * std::pow(d[7], 2) * d[12] + std::pow(d[0], 2) * std::pow(d[4], 2) * d[13] + 2 * d[0] * d[1] * d[4] * d[5] * d[13] + std::pow(d[1], 2) * std::pow(d[5], 2) * d[13] - std::pow(d[2], 2) * std::pow(d[6], 2) * d[14] - 2 * d[2] * d[3] * d[6] * d[7] * d[14] - std::pow(d[3], 2) * std::pow(d[7], 2) * d[14] + std::pow(d[0], 2) * std::pow(d[4], 2) * d[15] + 2 * d[0] * d[1] * d[4] * d[5] * d[15] + std::pow(d[1], 2) * std::pow(d[5], 2) * d[15] - 2 * d[2] * d[6] * d[12] * d[14] - 2 * d[3] * d[7] * d[12] * d[14] + 2 * d[0] * d[4] * d[13] * d[15] + 2 * d[1] * d[5] * d[13] * d[15];
					coeffs[15] = -std::pow(d[2], 2) * std::pow(d[6], 2) * d[12] * d[14] - 2 * d[2] * d[3] * d[6] * d[7] * d[12] * d[14] - std::pow(d[3], 2) * std::pow(d[7], 2) * d[12] * d[14] + std::pow(d[0], 2) * std::pow(d[4], 2) * d[13] * d[15] + 2 * d[0] * d[1] * d[4] * d[5] * d[13] * d[15] + std::pow(d[1], 2) * std::pow(d[5], 2) * d[13] * d[15];
					coeffs[16] = -2 * d[2] * d[10] * std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) * d[17] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) * d[17] + 2 * d[0] * d[8] * d[12] * std::pow(d[13], 2) * d[16] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[12] * std::pow(d[13], 2) * d[16] * std::pow(d[17], 2) + std::pow(d[12], 2) * std::pow(d[13], 2) * std::pow(d[16], 2) * d[17] - std::pow(d[12], 2) * std::pow(d[13], 2) * d[16] * std::pow(d[17], 2) + std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) * std::pow(d[17], 2) - d[12] * std::pow(d[13], 2) * std::pow(d[16], 2) * std::pow(d[17], 2);
					coeffs[17] = -2 * d[2] * d[10] * std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) - 4 * d[2] * d[10] * std::pow(d[12], 2) * d[13] * d[16] * d[17] - 4 * d[3] * d[11] * std::pow(d[12], 2) * d[13] * d[16] * d[17] + 4 * d[0] * d[8] * d[12] * std::pow(d[13], 2) * d[16] * d[17] + 4 * d[1] * d[9] * d[12] * std::pow(d[13], 2) * d[16] * d[17] - 2 * d[2] * d[10] * std::pow(d[12], 2) * std::pow(d[16], 2) * d[17] - 2 * d[3] * d[11] * std::pow(d[12], 2) * std::pow(d[16], 2) * d[17] - 4 * d[2] * d[10] * d[12] * d[13] * std::pow(d[16], 2) * d[17] - 4 * d[3] * d[11] * d[12] * d[13] * std::pow(d[16], 2) * d[17] + 2 * d[0] * d[8] * d[12] * std::pow(d[13], 2) * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[12] * std::pow(d[13], 2) * std::pow(d[17], 2) + 4 * d[0] * d[8] * d[12] * d[13] * d[16] * std::pow(d[17], 2) + 4 * d[1] * d[9] * d[12] * d[13] * d[16] * std::pow(d[17], 2) + 2 * d[0] * d[8] * std::pow(d[13], 2) * d[16] * std::pow(d[17], 2) + 2 * d[1] * d[9] * std::pow(d[13], 2) * d[16] * std::pow(d[17], 2) + 4 * std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) * d[17] - 4 * d[12] * std::pow(d[13], 2) * d[16] * std::pow(d[17], 2);
					coeffs[18] = -4 * d[2] * d[10] * std::pow(d[12], 2) * d[13] * d[16] - 4 * d[3] * d[11] * std::pow(d[12], 2) * d[13] * d[16] + 2 * d[0] * d[8] * d[12] * std::pow(d[13], 2) * d[16] + 2 * d[1] * d[9] * d[12] * std::pow(d[13], 2) * d[16] - 2 * d[2] * d[10] * std::pow(d[12], 2) * std::pow(d[16], 2) - 2 * d[3] * d[11] * std::pow(d[12], 2) * std::pow(d[16], 2) - 4 * d[2] * d[10] * d[12] * d[13] * std::pow(d[16], 2) - 4 * d[3] * d[11] * d[12] * d[13] * std::pow(d[16], 2) - 2 * d[2] * d[10] * std::pow(d[12], 2) * d[13] * d[17] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[13] * d[17] + 4 * d[0] * d[8] * d[12] * std::pow(d[13], 2) * d[17] + 4 * d[1] * d[9] * d[12] * std::pow(d[13], 2) * d[17] - 4 * d[2] * d[10] * std::pow(d[12], 2) * d[16] * d[17] - 4 * d[3] * d[11] * std::pow(d[12], 2) * d[16] * d[17] + 8 * d[0] * d[8] * d[12] * d[13] * d[16] * d[17] + 8 * d[1] * d[9] * d[12] * d[13] * d[16] * d[17] - 8 * d[2] * d[10] * d[12] * d[13] * d[16] * d[17] - 8 * d[3] * d[11] * d[12] * d[13] * d[16] * d[17] + 4 * d[0] * d[8] * std::pow(d[13], 2) * d[16] * d[17] + 4 * d[1] * d[9] * std::pow(d[13], 2) * d[16] * d[17] - 4 * d[2] * d[10] * d[12] * std::pow(d[16], 2) * d[17] - 4 * d[3] * d[11] * d[12] * std::pow(d[16], 2) * d[17] - 2 * d[2] * d[10] * d[13] * std::pow(d[16], 2) * d[17] - 2 * d[3] * d[11] * d[13] * std::pow(d[16], 2) * d[17] + 4 * d[0] * d[8] * d[12] * d[13] * std::pow(d[17], 2) + 4 * d[1] * d[9] * d[12] * d[13] * std::pow(d[17], 2) + 2 * d[0] * d[8] * std::pow(d[13], 2) * std::pow(d[17], 2) + 2 * d[1] * d[9] * std::pow(d[13], 2) * std::pow(d[17], 2) + 2 * d[0] * d[8] * d[12] * d[16] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[12] * d[16] * std::pow(d[17], 2) + 4 * d[0] * d[8] * d[13] * d[16] * std::pow(d[17], 2) + 4 * d[1] * d[9] * d[13] * d[16] * std::pow(d[17], 2) - std::pow(d[12], 2) * std::pow(d[13], 2) * d[16] + std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) - d[12] * std::pow(d[13], 2) * std::pow(d[16], 2) + std::pow(d[12], 2) * std::pow(d[13], 2) * d[17] + 4 * std::pow(d[12], 2) * d[13] * d[16] * d[17] - 4 * d[12] * std::pow(d[13], 2) * d[16] * d[17] + std::pow(d[12], 2) * std::pow(d[16], 2) * d[17] + 4 * d[12] * d[13] * std::pow(d[16], 2) * d[17] + std::pow(d[13], 2) * std::pow(d[16], 2) * d[17] + std::pow(d[12], 2) * d[13] * std::pow(d[17], 2) - d[12] * std::pow(d[13], 2) * std::pow(d[17], 2) - std::pow(d[12], 2) * d[16] * std::pow(d[17], 2) - 4 * d[12] * d[13] * d[16] * std::pow(d[17], 2) - std::pow(d[13], 2) * d[16] * std::pow(d[17], 2) - d[12] * std::pow(d[16], 2) * std::pow(d[17], 2) + d[13] * std::pow(d[16], 2) * std::pow(d[17], 2);
					coeffs[19] = -2 * d[2] * d[10] * std::pow(d[12], 2) * d[13] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[13] + 2 * d[0] * d[8] * d[12] * std::pow(d[13], 2) + 2 * d[1] * d[9] * d[12] * std::pow(d[13], 2) - 4 * d[2] * d[10] * std::pow(d[12], 2) * d[16] - 4 * d[3] * d[11] * std::pow(d[12], 2) * d[16] + 4 * d[0] * d[8] * d[12] * d[13] * d[16] + 4 * d[1] * d[9] * d[12] * d[13] * d[16] - 8 * d[2] * d[10] * d[12] * d[13] * d[16] - 8 * d[3] * d[11] * d[12] * d[13] * d[16] + 2 * d[0] * d[8] * std::pow(d[13], 2) * d[16] + 2 * d[1] * d[9] * std::pow(d[13], 2) * d[16] - 4 * d[2] * d[10] * d[12] * std::pow(d[16], 2) - 4 * d[3] * d[11] * d[12] * std::pow(d[16], 2) - 2 * d[2] * d[10] * d[13] * std::pow(d[16], 2) - 2 * d[3] * d[11] * d[13] * std::pow(d[16], 2) - 2 * d[2] * d[10] * std::pow(d[12], 2) * d[17] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[17] + 8 * d[0] * d[8] * d[12] * d[13] * d[17] + 8 * d[1] * d[9] * d[12] * d[13] * d[17] - 4 * d[2] * d[10] * d[12] * d[13] * d[17] - 4 * d[3] * d[11] * d[12] * d[13] * d[17] + 4 * d[0] * d[8] * std::pow(d[13], 2) * d[17] + 4 * d[1] * d[9] * std::pow(d[13], 2) * d[17] + 4 * d[0] * d[8] * d[12] * d[16] * d[17] + 4 * d[1] * d[9] * d[12] * d[16] * d[17] - 8 * d[2] * d[10] * d[12] * d[16] * d[17] - 8 * d[3] * d[11] * d[12] * d[16] * d[17] + 8 * d[0] * d[8] * d[13] * d[16] * d[17] + 8 * d[1] * d[9] * d[13] * d[16] * d[17] - 4 * d[2] * d[10] * d[13] * d[16] * d[17] - 4 * d[3] * d[11] * d[13] * d[16] * d[17] - 2 * d[2] * d[10] * std::pow(d[16], 2) * d[17] - 2 * d[3] * d[11] * std::pow(d[16], 2) * d[17] + 2 * d[0] * d[8] * d[12] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[12] * std::pow(d[17], 2) + 4 * d[0] * d[8] * d[13] * std::pow(d[17], 2) + 4 * d[1] * d[9] * d[13] * std::pow(d[17], 2) + 2 * d[0] * d[8] * d[16] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[16] * std::pow(d[17], 2) - 4 * d[12] * std::pow(d[13], 2) * d[16] + 4 * std::pow(d[12], 2) * d[13] * d[17] + 4 * d[13] * std::pow(d[16], 2) * d[17] - 4 * d[12] * d[16] * std::pow(d[17], 2);
					coeffs[20] = -std::pow(d[2], 2) * std::pow(d[10], 2) * std::pow(d[12], 2) * std::pow(d[16], 2) - 2 * d[2] * d[3] * d[10] * d[11] * std::pow(d[12], 2) * std::pow(d[16], 2) - std::pow(d[3], 2) * std::pow(d[11], 2) * std::pow(d[12], 2) * std::pow(d[16], 2) + std::pow(d[0], 2) * std::pow(d[8], 2) * std::pow(d[13], 2) * std::pow(d[17], 2) + 2 * d[0] * d[1] * d[8] * d[9] * std::pow(d[13], 2) * std::pow(d[17], 2) + std::pow(d[1], 2) * std::pow(d[9], 2) * std::pow(d[13], 2) * std::pow(d[17], 2) - 2 * d[2] * d[10] * std::pow(d[12], 2) * d[13] * d[16] * d[17] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[13] * d[16] * d[17] + 2 * d[0] * d[8] * d[12] * std::pow(d[13], 2) * d[16] * d[17] + 2 * d[1] * d[9] * d[12] * std::pow(d[13], 2) * d[16] * d[17] - 2 * d[2] * d[10] * d[12] * d[13] * std::pow(d[16], 2) * d[17] - 2 * d[3] * d[11] * d[12] * d[13] * std::pow(d[16], 2) * d[17] + 2 * d[0] * d[8] * d[12] * d[13] * d[16] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[12] * d[13] * d[16] * std::pow(d[17], 2) + std::pow(d[12], 2) * d[13] * std::pow(d[16], 2) * d[17] - d[12] * std::pow(d[13], 2) * d[16] * std::pow(d[17], 2);
					coeffs[21] = -2 * d[2] * d[10] * std::pow(d[12], 2) - 2 * d[3] * d[11] * std::pow(d[12], 2) + 4 * d[0] * d[8] * d[12] * d[13] + 4 * d[1] * d[9] * d[12] * d[13] - 4 * d[2] * d[10] * d[12] * d[13] - 4 * d[3] * d[11] * d[12] * d[13] + 2 * d[0] * d[8] * std::pow(d[13], 2) + 2 * d[1] * d[9] * std::pow(d[13], 2) + 2 * d[0] * d[8] * d[12] * d[16] + 2 * d[1] * d[9] * d[12] * d[16] - 8 * d[2] * d[10] * d[12] * d[16] - 8 * d[3] * d[11] * d[12] * d[16] + 4 * d[0] * d[8] * d[13] * d[16] + 4 * d[1] * d[9] * d[13] * d[16] - 4 * d[2] * d[10] * d[13] * d[16] - 4 * d[3] * d[11] * d[13] * d[16] - 2 * d[2] * d[10] * std::pow(d[16], 2) - 2 * d[3] * d[11] * std::pow(d[16], 2) + 4 * d[0] * d[8] * d[12] * d[17] + 4 * d[1] * d[9] * d[12] * d[17] - 4 * d[2] * d[10] * d[12] * d[17] - 4 * d[3] * d[11] * d[12] * d[17] + 8 * d[0] * d[8] * d[13] * d[17] + 8 * d[1] * d[9] * d[13] * d[17] - 2 * d[2] * d[10] * d[13] * d[17] - 2 * d[3] * d[11] * d[13] * d[17] + 4 * d[0] * d[8] * d[16] * d[17] + 4 * d[1] * d[9] * d[16] * d[17] - 4 * d[2] * d[10] * d[16] * d[17] - 4 * d[3] * d[11] * d[16] * d[17] + 2 * d[0] * d[8] * std::pow(d[17], 2) + 2 * d[1] * d[9] * std::pow(d[17], 2) + std::pow(d[12], 2) * d[13] - d[12] * std::pow(d[13], 2) - std::pow(d[12], 2) * d[16] - 4 * d[12] * d[13] * d[16] - std::pow(d[13], 2) * d[16] - d[12] * std::pow(d[16], 2) + d[13] * std::pow(d[16], 2) + std::pow(d[12], 2) * d[17] + 4 * d[12] * d[13] * d[17] + std::pow(d[13], 2) * d[17] - 4 * d[12] * d[16] * d[17] + 4 * d[13] * d[16] * d[17] + std::pow(d[16], 2) * d[17] - d[12] * std::pow(d[17], 2) + d[13] * std::pow(d[17], 2) - d[16] * std::pow(d[17], 2);
					coeffs[22] = -2 * std::pow(d[2], 2) * std::pow(d[10], 2) * std::pow(d[12], 2) * d[16] - 4 * d[2] * d[3] * d[10] * d[11] * std::pow(d[12], 2) * d[16] - 2 * std::pow(d[3], 2) * std::pow(d[11], 2) * std::pow(d[12], 2) * d[16] - 2 * std::pow(d[2], 2) * std::pow(d[10], 2) * d[12] * std::pow(d[16], 2) - 4 * d[2] * d[3] * d[10] * d[11] * d[12] * std::pow(d[16], 2) - 2 * std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] * std::pow(d[16], 2) + 2 * std::pow(d[0], 2) * std::pow(d[8], 2) * std::pow(d[13], 2) * d[17] + 4 * d[0] * d[1] * d[8] * d[9] * std::pow(d[13], 2) * d[17] + 2 * std::pow(d[1], 2) * std::pow(d[9], 2) * std::pow(d[13], 2) * d[17] + 2 * std::pow(d[0], 2) * std::pow(d[8], 2) * d[13] * std::pow(d[17], 2) + 4 * d[0] * d[1] * d[8] * d[9] * d[13] * std::pow(d[17], 2) + 2 * std::pow(d[1], 2) * std::pow(d[9], 2) * d[13] * std::pow(d[17], 2) - 2 * d[2] * d[10] * std::pow(d[12], 2) * d[13] * d[16] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[13] * d[16] - 2 * d[2] * d[10] * d[12] * d[13] * std::pow(d[16], 2) - 2 * d[3] * d[11] * d[12] * d[13] * std::pow(d[16], 2) + 2 * d[0] * d[8] * d[12] * std::pow(d[13], 2) * d[17] + 2 * d[1] * d[9] * d[12] * std::pow(d[13], 2) * d[17] - 2 * d[2] * d[10] * std::pow(d[12], 2) * d[16] * d[17] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[16] * d[17] + 8 * d[0] * d[8] * d[12] * d[13] * d[16] * d[17] + 8 * d[1] * d[9] * d[12] * d[13] * d[16] * d[17] - 8 * d[2] * d[10] * d[12] * d[13] * d[16] * d[17] - 8 * d[3] * d[11] * d[12] * d[13] * d[16] * d[17] + 2 * d[0] * d[8] * std::pow(d[13], 2) * d[16] * d[17] + 2 * d[1] * d[9] * std::pow(d[13], 2) * d[16] * d[17] - 2 * d[2] * d[10] * d[12] * std::pow(d[16], 2) * d[17] - 2 * d[3] * d[11] * d[12] * std::pow(d[16], 2) * d[17] + 2 * d[0] * d[8] * d[12] * d[13] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[12] * d[13] * std::pow(d[17], 2) + 2 * d[0] * d[8] * d[13] * d[16] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[13] * d[16] * std::pow(d[17], 2) + 2 * std::pow(d[12], 2) * d[13] * d[16] * d[17] - 2 * d[12] * std::pow(d[13], 2) * d[16] * d[17] + 2 * d[12] * d[13] * std::pow(d[16], 2) * d[17] - 2 * d[12] * d[13] * d[16] * std::pow(d[17], 2);
					coeffs[23] = 2 * d[0] * d[8] * d[12] + 2 * d[1] * d[9] * d[12] - 4 * d[2] * d[10] * d[12] - 4 * d[3] * d[11] * d[12] + 4 * d[0] * d[8] * d[13] + 4 * d[1] * d[9] * d[13] - 2 * d[2] * d[10] * d[13] - 2 * d[3] * d[11] * d[13] + 2 * d[0] * d[8] * d[16] + 2 * d[1] * d[9] * d[16] - 4 * d[2] * d[10] * d[16] - 4 * d[3] * d[11] * d[16] + 4 * d[0] * d[8] * d[17] + 4 * d[1] * d[9] * d[17] - 2 * d[2] * d[10] * d[17] - 2 * d[3] * d[11] * d[17] - 4 * d[12] * d[16] + 4 * d[13] * d[17];
					coeffs[24] = -std::pow(d[2], 2) * std::pow(d[10], 2) * std::pow(d[12], 2) - 2 * d[2] * d[3] * d[10] * d[11] * std::pow(d[12], 2) - std::pow(d[3], 2) * std::pow(d[11], 2) * std::pow(d[12], 2) + std::pow(d[0], 2) * std::pow(d[8], 2) * std::pow(d[13], 2) + 2 * d[0] * d[1] * d[8] * d[9] * std::pow(d[13], 2) + std::pow(d[1], 2) * std::pow(d[9], 2) * std::pow(d[13], 2) - 4 * std::pow(d[2], 2) * std::pow(d[10], 2) * d[12] * d[16] - 8 * d[2] * d[3] * d[10] * d[11] * d[12] * d[16] - 4 * std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] * d[16] - std::pow(d[2], 2) * std::pow(d[10], 2) * std::pow(d[16], 2) - 2 * d[2] * d[3] * d[10] * d[11] * std::pow(d[16], 2) - std::pow(d[3], 2) * std::pow(d[11], 2) * std::pow(d[16], 2) + 4 * std::pow(d[0], 2) * std::pow(d[8], 2) * d[13] * d[17] + 8 * d[0] * d[1] * d[8] * d[9] * d[13] * d[17] + 4 * std::pow(d[1], 2) * std::pow(d[9], 2) * d[13] * d[17] + std::pow(d[0], 2) * std::pow(d[8], 2) * std::pow(d[17], 2) + 2 * d[0] * d[1] * d[8] * d[9] * std::pow(d[17], 2) + std::pow(d[1], 2) * std::pow(d[9], 2) * std::pow(d[17], 2) - 2 * d[2] * d[10] * std::pow(d[12], 2) * d[16] - 2 * d[3] * d[11] * std::pow(d[12], 2) * d[16] + 2 * d[0] * d[8] * d[12] * d[13] * d[16] + 2 * d[1] * d[9] * d[12] * d[13] * d[16] - 8 * d[2] * d[10] * d[12] * d[13] * d[16] - 8 * d[3] * d[11] * d[12] * d[13] * d[16] - 2 * d[2] * d[10] * d[12] * std::pow(d[16], 2) - 2 * d[3] * d[11] * d[12] * std::pow(d[16], 2) + 8 * d[0] * d[8] * d[12] * d[13] * d[17] + 8 * d[1] * d[9] * d[12] * d[13] * d[17] - 2 * d[2] * d[10] * d[12] * d[13] * d[17] - 2 * d[3] * d[11] * d[12] * d[13] * d[17] + 2 * d[0] * d[8] * std::pow(d[13], 2) * d[17] + 2 * d[1] * d[9] * std::pow(d[13], 2) * d[17] + 2 * d[0] * d[8] * d[12] * d[16] * d[17] + 2 * d[1] * d[9] * d[12] * d[16] * d[17] - 8 * d[2] * d[10] * d[12] * d[16] * d[17] - 8 * d[3] * d[11] * d[12] * d[16] * d[17] + 8 * d[0] * d[8] * d[13] * d[16] * d[17] + 8 * d[1] * d[9] * d[13] * d[16] * d[17] - 2 * d[2] * d[10] * d[13] * d[16] * d[17] - 2 * d[3] * d[11] * d[13] * d[16] * d[17] + 2 * d[0] * d[8] * d[13] * std::pow(d[17], 2) + 2 * d[1] * d[9] * d[13] * std::pow(d[17], 2) - d[12] * std::pow(d[13], 2) * d[16] + std::pow(d[12], 2) * d[13] * d[17] + d[13] * std::pow(d[16], 2) * d[17] - d[12] * d[16] * std::pow(d[17], 2);
					coeffs[25] = 2 * d[0] * d[8] + 2 * d[1] * d[9] - 2 * d[2] * d[10] - 2 * d[3] * d[11] - d[12] + d[13] - d[16] + d[17];
					coeffs[26] = -2 * std::pow(d[2], 2) * std::pow(d[10], 2) * d[12] - 4 * d[2] * d[3] * d[10] * d[11] * d[12] - 2 * std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] + 2 * std::pow(d[0], 2) * std::pow(d[8], 2) * d[13] + 4 * d[0] * d[1] * d[8] * d[9] * d[13] + 2 * std::pow(d[1], 2) * std::pow(d[9], 2) * d[13] - 2 * std::pow(d[2], 2) * std::pow(d[10], 2) * d[16] - 4 * d[2] * d[3] * d[10] * d[11] * d[16] - 2 * std::pow(d[3], 2) * std::pow(d[11], 2) * d[16] + 2 * std::pow(d[0], 2) * std::pow(d[8], 2) * d[17] + 4 * d[0] * d[1] * d[8] * d[9] * d[17] + 2 * std::pow(d[1], 2) * std::pow(d[9], 2) * d[17] + 2 * d[0] * d[8] * d[12] * d[13] + 2 * d[1] * d[9] * d[12] * d[13] - 2 * d[2] * d[10] * d[12] * d[13] - 2 * d[3] * d[11] * d[12] * d[13] - 8 * d[2] * d[10] * d[12] * d[16] - 8 * d[3] * d[11] * d[12] * d[16] + 2 * d[0] * d[8] * d[13] * d[16] + 2 * d[1] * d[9] * d[13] * d[16] - 2 * d[2] * d[10] * d[13] * d[16] - 2 * d[3] * d[11] * d[13] * d[16] + 2 * d[0] * d[8] * d[12] * d[17] + 2 * d[1] * d[9] * d[12] * d[17] - 2 * d[2] * d[10] * d[12] * d[17] - 2 * d[3] * d[11] * d[12] * d[17] + 8 * d[0] * d[8] * d[13] * d[17] + 8 * d[1] * d[9] * d[13] * d[17] + 2 * d[0] * d[8] * d[16] * d[17] + 2 * d[1] * d[9] * d[16] * d[17] - 2 * d[2] * d[10] * d[16] * d[17] - 2 * d[3] * d[11] * d[16] * d[17] - 2 * d[12] * d[13] * d[16] + 2 * d[12] * d[13] * d[17] - 2 * d[12] * d[16] * d[17] + 2 * d[13] * d[16] * d[17];
					coeffs[27] = -std::pow(d[2], 2) * std::pow(d[10], 2) * std::pow(d[12], 2) * d[16] - 2 * d[2] * d[3] * d[10] * d[11] * std::pow(d[12], 2) * d[16] - std::pow(d[3], 2) * std::pow(d[11], 2) * std::pow(d[12], 2) * d[16] - std::pow(d[2], 2) * std::pow(d[10], 2) * d[12] * std::pow(d[16], 2) - 2 * d[2] * d[3] * d[10] * d[11] * d[12] * std::pow(d[16], 2) - std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] * std::pow(d[16], 2) + std::pow(d[0], 2) * std::pow(d[8], 2) * std::pow(d[13], 2) * d[17] + 2 * d[0] * d[1] * d[8] * d[9] * std::pow(d[13], 2) * d[17] + std::pow(d[1], 2) * std::pow(d[9], 2) * std::pow(d[13], 2) * d[17] + std::pow(d[0], 2) * std::pow(d[8], 2) * d[13] * std::pow(d[17], 2) + 2 * d[0] * d[1] * d[8] * d[9] * d[13] * std::pow(d[17], 2) + std::pow(d[1], 2) * std::pow(d[9], 2) * d[13] * std::pow(d[17], 2) + 2 * d[0] * d[8] * d[12] * d[13] * d[16] * d[17] + 2 * d[1] * d[9] * d[12] * d[13] * d[16] * d[17] - 2 * d[2] * d[10] * d[12] * d[13] * d[16] * d[17] - 2 * d[3] * d[11] * d[12] * d[13] * d[16] * d[17];
					coeffs[28] = std::pow(d[0], 2) * std::pow(d[8], 2) + 2 * d[0] * d[1] * d[8] * d[9] + std::pow(d[1], 2) * std::pow(d[9], 2) - std::pow(d[2], 2) * std::pow(d[10], 2) - 2 * d[2] * d[3] * d[10] * d[11] - std::pow(d[3], 2) * std::pow(d[11], 2) - 2 * d[2] * d[10] * d[12] - 2 * d[3] * d[11] * d[12] + 2 * d[0] * d[8] * d[13] + 2 * d[1] * d[9] * d[13] - 2 * d[2] * d[10] * d[16] - 2 * d[3] * d[11] * d[16] + 2 * d[0] * d[8] * d[17] + 2 * d[1] * d[9] * d[17] - d[12] * d[16] + d[13] * d[17];
					coeffs[29] = -4 * std::pow(d[2], 2) * std::pow(d[10], 2) * d[12] * d[16] - 8 * d[2] * d[3] * d[10] * d[11] * d[12] * d[16] - 4 * std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] * d[16] + 4 * std::pow(d[0], 2) * std::pow(d[8], 2) * d[13] * d[17] + 8 * d[0] * d[1] * d[8] * d[9] * d[13] * d[17] + 4 * std::pow(d[1], 2) * std::pow(d[9], 2) * d[13] * d[17] - 2 * d[2] * d[10] * d[12] * d[13] * d[16] - 2 * d[3] * d[11] * d[12] * d[13] * d[16] + 2 * d[0] * d[8] * d[12] * d[13] * d[17] + 2 * d[1] * d[9] * d[12] * d[13] * d[17] - 2 * d[2] * d[10] * d[12] * d[16] * d[17] - 2 * d[3] * d[11] * d[12] * d[16] * d[17] + 2 * d[0] * d[8] * d[13] * d[16] * d[17] + 2 * d[1] * d[9] * d[13] * d[16] * d[17];
					coeffs[30] = -std::pow(d[2], 2) * std::pow(d[10], 2) * d[12] - 2 * d[2] * d[3] * d[10] * d[11] * d[12] - std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] + std::pow(d[0], 2) * std::pow(d[8], 2) * d[13] + 2 * d[0] * d[1] * d[8] * d[9] * d[13] + std::pow(d[1], 2) * std::pow(d[9], 2) * d[13] - std::pow(d[2], 2) * std::pow(d[10], 2) * d[16] - 2 * d[2] * d[3] * d[10] * d[11] * d[16] - std::pow(d[3], 2) * std::pow(d[11], 2) * d[16] + std::pow(d[0], 2) * std::pow(d[8], 2) * d[17] + 2 * d[0] * d[1] * d[8] * d[9] * d[17] + std::pow(d[1], 2) * std::pow(d[9], 2) * d[17] - 2 * d[2] * d[10] * d[12] * d[16] - 2 * d[3] * d[11] * d[12] * d[16] + 2 * d[0] * d[8] * d[13] * d[17] + 2 * d[1] * d[9] * d[13] * d[17];
					coeffs[31] = -std::pow(d[2], 2) * std::pow(d[10], 2) * d[12] * d[16] - 2 * d[2] * d[3] * d[10] * d[11] * d[12] * d[16] - std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] * d[16] + std::pow(d[0], 2) * std::pow(d[8], 2) * d[13] * d[17] + 2 * d[0] * d[1] * d[8] * d[9] * d[13] * d[17] + std::pow(d[1], 2) * std::pow(d[9], 2) * d[13] * d[17];

					MatrixXd C0 = MatrixXd::Zero(6, 6);
					C0 << coeffs[15], coeffs[14], coeffs[12], coeffs[9], 0, 0,
						coeffs[31], coeffs[30], coeffs[28], coeffs[25], 0, 0,
						0, coeffs[15], coeffs[14], coeffs[12], coeffs[9], 0,
						0, coeffs[31], coeffs[30], coeffs[28], coeffs[25], 0,
						0, 0, coeffs[15], coeffs[14], coeffs[12], coeffs[9],
						0, 0, coeffs[31], coeffs[30], coeffs[28], coeffs[25];

					MatrixXd C1 = MatrixXd::Zero(6, 5);
					C1 << coeffs[13], coeffs[10], coeffs[7], 0, 0,
						coeffs[29], coeffs[26], coeffs[23], 0, 0,
						0, coeffs[13], coeffs[10], coeffs[7], 0,
						0, coeffs[29], coeffs[26], coeffs[23], 0,
						0, 0, coeffs[13], coeffs[10], coeffs[7],
						0, 0, coeffs[29], coeffs[26], coeffs[23];

					MatrixXd C2 = MatrixXd::Zero(6, 5);
					C2 << coeffs[11], coeffs[8], coeffs[5], 0, 0,
						coeffs[27], coeffs[24], coeffs[21], 0, 0,
						0, coeffs[11], coeffs[8], coeffs[5], 0,
						0, coeffs[27], coeffs[24], coeffs[21], 0,
						0, 0, coeffs[11], coeffs[8], coeffs[5],
						0, 0, coeffs[27], coeffs[24], coeffs[21];

					MatrixXd C3 = MatrixXd::Zero(6, 4);
					C3 << coeffs[6], coeffs[3], 0, 0,
						coeffs[22], coeffs[19], 0, 0,
						0, coeffs[6], coeffs[3], 0,
						0, coeffs[22], coeffs[19], 0,
						0, 0, coeffs[6], coeffs[3],
						0, 0, coeffs[22], coeffs[19];

					MatrixXd C4 = MatrixXd::Zero(6, 4);
					C4 << coeffs[4], coeffs[2], 0, 0,
						coeffs[20], coeffs[18], 0, 0,
						0, coeffs[4], coeffs[2], 0,
						0, coeffs[20], coeffs[18], 0,
						0, 0, coeffs[4], coeffs[2],
						0, 0, coeffs[20], coeffs[18];

					MatrixXd C5 = MatrixXd::Zero(6, 3);
					C5 << coeffs[1], 0, 0,
						coeffs[17], 0, 0,
						0, coeffs[1], 0,
						0, coeffs[17], 0,
						0, 0, coeffs[1],
						0, 0, coeffs[17];

					MatrixXd C6 = MatrixXd::Zero(6, 3);
					C6 << coeffs[0], 0, 0,
						coeffs[16], 0, 0,
						0, coeffs[0], 0,
						0, coeffs[16], 0,
						0, 0, coeffs[0],
						0, 0, coeffs[16];

					MatrixXd M(6, 24);
					M << C6, C5, C4, C3, C2, C1;
					M = (-C0.fullPivLu().solve(M)).eval();

					MatrixXd K(24, 24);
					K.setZero();
					K(0, 3) = 1;
					K(1, 4) = 1;
					K(2, 5) = 1;
					K(3, 7) = 1;
					K(4, 8) = 1;
					K(5, 9) = 1;
					K(6, 10) = 1;
					K(7, 11) = 1;
					K(8, 12) = 1;
					K(9, 13) = 1;
					K(10, 15) = 1;
					K(11, 16) = 1;
					K(12, 17) = 1;
					K(13, 18) = 1;
					K(14, 19) = 1;
					K(15, 20) = 1;
					K(16, 21) = 1;
					K(17, 22) = 1;
					K(18, 23) = 1;

					K.block<5, 24>(19, 0) = M.block<5, 24>(1, 0);




					EigenSolver<MatrixXd> es(K);
					ArrayXcd D = es.eigenvalues();
					ArrayXXcd V = es.eigenvectors();

					MatrixXcd sols(2, 24);
					sols.row(0) = V.row(2)/V.row(1);
					sols.row(1) = 1./D.transpose();

					for (size_t k = 0; k < 24; ++k)
					{
						if (sols(0, k).imag() > std::numeric_limits<double>::epsilon() || sols(0, k).real() < 0)
							continue;

						double focal = sqrt(sols(0, k).real());
						Eigen::Vector3d u11(u1 / (focal * (1 + sols(1, k).real() * r1)), v1 / (focal * (1 + sols(1, k).real() * r1)), 1);
						Eigen::Vector3d u21(u2 / (focal * (1 + sols(1, k).real() * r2)), v2 / (focal * (1 + sols(1, k).real() * r2)), 1);

						Eigen::Vector3d u12(u3 / (focal * (1 + sols(1, k).real() * r3)), v3 / (focal * (1 + sols(1, k).real() * r3)), 1);
						Eigen::Vector3d u22(u4 / (focal * (1 + sols(1, k).real() * r4)), v4 / (focal * (1 + sols(1, k).real() * r4)), 1);

						Eigen::Vector3d u13(u5 / (focal * (1 + sols(1, k).real() * r5)), v5 / (focal * (1 + sols(1, k).real() * r5)), 1);
						Eigen::Vector3d u23(u6 / (focal * (1 + sols(1, k).real() * r6)), v6 / (focal * (1 + sols(1, k).real() * r6)), 1);

						u11.normalize();
						u21.normalize();
						u12.normalize();
						u22.normalize();
						u13.normalize();
						u23.normalize();

						Eigen::Matrix3d C = u21 * u11.transpose() + u22 * u12.transpose() + u23 * u13.transpose();
						JacobiSVD<MatrixXd> svd(C, ComputeFullU | ComputeFullV);
						Eigen::Matrix3d U = svd.matrixU();
						Eigen::Matrix3d V = svd.matrixV().transpose();

						Eigen::Matrix3d S;
						S << 1, 0, 0,
							0, 1, 0,
							0, 0, ((U * V).determinant() > 0) ? 1 : -1;

						Eigen::Matrix3d R = U * S * V;
						Eigen::Matrix3d K2, K1;
						K2 << focal, 0, 0,
							0, focal, 0,
							0, 0, 1;
						K1 << focal, 0, 0,
							0, focal, 0,
							0, 0, 1;
						Eigen::Matrix3d H = K2 * R * K1.inverse();

						if (H.hasNaN())
							continue;

						RadialHomography model;
						model.descriptor.block<3, 3>(0, 0) = H;
						model.descriptor(0, 3) = sols(1, k).real();
						model.descriptor(1, 3) = sols(1, k).real();
						model.descriptor(2, 3) = focal;
						models_.push_back(model);
					}
				}

				return models_.size() > 0;
			} // namespace solver

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac

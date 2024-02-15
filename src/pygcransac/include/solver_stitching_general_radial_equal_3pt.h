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

				static const int coeffs_ind[] = {0, 16, 1, 0, 16, 17, 2, 1, 0, 16, 17, 18, 3, 2, 1, 0, 16, 17, 18, 19, 4, 0, 16, 20, 5, 3, 2, 1, 0, 16, 17, 18, 19, 21, 6, 4, 1, 0, 17, 16, 20, 22, 7, 5, 3, 2, 1, 0, 16, 17,
												 18, 19, 21, 23, 8, 6, 4, 2, 1, 0, 18, 17, 16, 20, 22, 24, 9, 7, 5, 3, 2, 1, 0, 16, 17, 18, 19, 21, 23, 25, 10, 8, 6, 4, 3, 2, 1, 0, 16, 19, 18, 17, 20, 22, 24, 26, 11, 4, 0, 16,
												 20, 27, 9, 7, 5, 3, 2, 1, 0, 16, 17, 18, 19, 21, 23, 25, 12, 10, 8, 6, 5, 4, 3, 2, 1, 0, 17, 21, 19, 16, 18, 20, 22, 24, 26, 28, 13, 11, 6, 4, 1, 0, 17, 16, 22, 20, 27, 29, 9, 7,
												 5, 3, 2, 1, 0, 16, 17, 18, 19, 21, 23, 25, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, 16, 18, 23, 21, 17, 20, 19, 22, 24, 26, 28, 14, 13, 11, 8, 6, 4, 2, 1, 0, 18, 17, 16, 24, 22, 20, 27,
												 29, 30, 9, 7, 5, 3, 2, 1, 0, 16, 17, 18, 19, 21, 23, 25, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 16, 17, 19, 25, 20, 23, 18, 22, 21, 24, 26, 28, 14, 13, 11, 10, 8, 6, 4, 3, 2, 1,
												 0, 19, 18, 17, 20, 26, 24, 16, 22, 27, 29, 30, 15, 11, 4, 20, 27, 31, 9, 7, 5, 3, 2, 1, 17, 18, 19, 21, 23, 25, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 16, 20, 17, 18, 21, 22, 25, 19,
												 24, 23, 26, 28, 14, 13, 12, 11, 10, 8, 6, 5, 4, 3, 2, 1, 21, 16, 19, 18, 22, 28, 26, 20, 0, 17, 24, 27, 29, 30, 15, 13, 11, 6, 4, 22, 20, 29, 27, 31, 9, 7, 5, 3, 2, 18, 19, 21, 23, 25,
												 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 20, 17, 22, 18, 19, 23, 24, 21, 26, 25, 28, 14, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 2, 23, 17, 21, 16, 19, 20, 24, 28, 22, 1, 18, 27, 26, 29, 0, 30,
												 15, 14, 13, 11, 8, 6, 4, 24, 22, 20, 30, 29, 27, 31, 9, 7, 5, 3, 19, 21, 23, 25, 12, 10, 9, 8, 7, 6, 5, 4, 3, 20, 2, 22, 18, 24, 19, 21, 25, 26, 23, 28, 14, 13, 12, 11, 10, 9, 8, 7,
												 6, 5, 4, 3, 25, 18, 23, 17, 21, 20, 22, 26, 27, 24, 2, 19, 29, 28, 30, 1, 15, 14, 13, 11, 10, 8, 6, 4, 26, 24, 22, 27, 30, 20, 29, 31, 15, 11, 27, 31, 9, 7, 5, 21, 23, 25, 12, 10, 9, 8,
												 7, 6, 5, 22, 3, 24, 19, 26, 21, 23, 28, 25, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 19, 25, 20, 18, 23, 27, 22, 24, 28, 29, 26, 3, 21, 30, 2, 15, 14, 13, 12, 11, 10, 8, 6, 28, 20, 26, 24,
												 29, 27, 4, 22, 30, 31, 15, 13, 11, 29, 27, 31, 9, 7, 23, 25, 12, 10, 9, 8, 7, 24, 5, 26, 21, 28, 23, 25, 9, 25, 12, 10, 9, 26, 7, 28, 23, 25, 14, 13, 12, 11, 10, 9, 27, 8, 23, 29, 24, 21,
												 30, 26, 28, 7, 25, 5, 14, 13, 12, 11, 10, 9, 8, 7, 6, 21, 27, 22, 19, 25, 29, 24, 26, 30, 28, 5, 23, 3, 15, 14, 13, 12, 11, 10, 8, 22, 28, 20, 26, 27, 30, 29, 6, 24, 31, 4, 15, 14, 13, 11,
												 30, 29, 27, 31, 31, 15, 15, 31, 30, 14, 15, 31, 14, 30, 28, 12, 14, 30, 12, 28, 25, 9, 12, 28, 9, 25, 14, 13, 12, 29, 10, 25, 30, 26, 23, 28, 9, 7, 15, 14, 13, 28, 31, 29, 26, 30, 12, 10, 15, 14,
												 13, 12, 11, 26, 27, 24, 31, 29, 30, 10, 28, 8, 15, 14, 13, 12, 11, 10, 24, 22, 28, 27, 29, 31, 30, 8, 26, 6, 15, 30, 29, 31, 14, 13, 15, 14, 29, 27, 31, 13, 30, 11, 15, 14, 13, 27, 30, 31, 11, 29,
												 15, 14, 13, 11, 30, 29, 31, 27, 31, 15, 15, 31, 15, 31, 15, 31, 15, 31};

				static const int C_ind[] = {0, 47, 48, 49, 94, 95, 96, 97, 98, 141, 142, 143, 144, 145, 146, 147, 188, 189, 190, 191, 192, 196, 226, 239, 240, 241, 242, 243, 245, 282, 284, 285, 286, 287, 288, 289, 292, 294, 322, 324, 334, 335, 336, 337, 338, 339, 341, 343, 376, 378,
											380, 381, 382, 383, 384, 385, 386, 388, 390, 392, 418, 420, 425, 429, 430, 431, 432, 433, 434, 435, 437, 439, 441, 467, 472, 474, 476, 477, 478, 479, 480, 481, 482, 483, 484, 486, 488, 490, 513, 514, 516, 521, 524, 525, 526, 527, 528, 532, 539, 549,
											562, 575, 577, 578, 579, 581, 583, 585, 588, 606, 611, 616, 618, 620, 621, 622, 624, 625, 626, 627, 628, 629, 630, 632, 634, 637, 657, 658, 660, 661, 665, 666, 668, 669, 670, 671, 672, 673, 676, 678, 683, 686, 693, 697, 706, 708, 718, 719, 722, 723,
											725, 727, 729, 732, 735, 746, 750, 755, 760, 762, 764, 765, 769, 770, 771, 772, 773, 774, 775, 776, 778, 781, 784, 800, 801, 802, 804, 805, 808, 809, 810, 812, 813, 814, 816, 817, 818, 820, 822, 824, 827, 830, 833, 837, 841, 845, 850, 852, 857, 861,
											862, 863, 867, 869, 871, 873, 876, 879, 882, 886, 890, 894, 899, 904, 906, 908, 914, 915, 916, 917, 918, 919, 920, 921, 922, 925, 928, 931, 943, 944, 945, 946, 947, 948, 949, 952, 953, 954, 956, 957, 961, 962, 963, 964, 966, 968, 970, 971, 974, 977,
											980, 981, 985, 989, 993, 994, 996, 999, 1001, 1004, 1005, 1006, 1008, 1012, 1019, 1029, 1042, 1055, 1061, 1063, 1065, 1068, 1071, 1074, 1078, 1082, 1086, 1091, 1096, 1098, 1107, 1109, 1110, 1111, 1112, 1113, 1114, 1116, 1117, 1120, 1123, 1127, 1131, 1134, 1135, 1136, 1137, 1139, 1140, 1141,
											1144, 1145, 1146, 1148, 1154, 1155, 1156, 1157, 1158, 1160, 1162, 1163, 1165, 1166, 1169, 1172, 1173, 1176, 1177, 1181, 1185, 1186, 1188, 1189, 1190, 1191, 1193, 1194, 1196, 1197, 1201, 1204, 1206, 1211, 1214, 1221, 1225, 1234, 1236, 1246, 1255, 1257, 1260, 1263, 1266, 1270, 1274, 1278, 1283, 1288,
											1301, 1303, 1304, 1305, 1306, 1308, 1309, 1311, 1312, 1315, 1319, 1322, 1323, 1326, 1327, 1328, 1329, 1331, 1333, 1336, 1337, 1338, 1347, 1349, 1350, 1351, 1352, 1354, 1355, 1357, 1358, 1360, 1361, 1364, 1365, 1368, 1369, 1372, 1373, 1376, 1377, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388,
											1394, 1396, 1398, 1400, 1403, 1406, 1409, 1413, 1417, 1421, 1426, 1428, 1433, 1437, 1449, 1452, 1455, 1458, 1462, 1466, 1470, 1475, 1495, 1497, 1498, 1500, 1501, 1503, 1504, 1506, 1507, 1510, 1511, 1514, 1515, 1518, 1519, 1520, 1521, 1523, 1525, 1528, 1541, 1543, 1544, 1545, 1546, 1547, 1549, 1550,
											1552, 1553, 1555, 1556, 1557, 1560, 1561, 1564, 1565, 1567, 1568, 1569, 1571, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1587, 1590, 1592, 1594, 1595, 1598, 1601, 1604, 1605, 1609, 1613, 1617, 1620, 1623, 1625, 1628, 1636, 1643, 1653, 1666, 1692, 1695, 1698, 1702, 1706, 1710, 1737, 1740, 1741, 1743,
											1744, 1746, 1747, 1750, 1751, 1754, 1755, 1758, 1759, 1760, 1763, 1765, 1783, 1785, 1786, 1788, 1789, 1790, 1792, 1793, 1795, 1796, 1799, 1800, 1801, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1811, 1813, 1814, 1815, 1816, 1819, 1829, 1832, 1834, 1835, 1837, 1838, 1841, 1844, 1845, 1848, 1849, 1853,
											1857, 1861, 1862, 1863, 1865, 1866, 1878, 1883, 1886, 1893, 1897, 1908, 1935, 1938, 1942, 1946, 1980, 1983, 1984, 1986, 1987, 1990, 1991, 1994, 1995, 1998, 1999, 2000, 2034, 2038, 2079, 2082, 2083, 2086, 2087, 2090, 2091, 2095, 2124, 2127, 2128, 2130, 2131, 2132, 2134, 2135, 2136, 2138, 2139, 2140,
											2142, 2143, 2144, 2150, 2151, 2155, 2169, 2172, 2173, 2175, 2176, 2177, 2179, 2180, 2183, 2184, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2195, 2197, 2198, 2199, 2203, 2215, 2218, 2221, 2222, 2224, 2225, 2228, 2232, 2233, 2236, 2237, 2240, 2241, 2245, 2246, 2247, 2248, 2251, 2264, 2267, 2270, 2273,
											2277, 2281, 2285, 2297, 2332, 2347, 2375, 2379, 2380, 2395, 2418, 2422, 2423, 2427, 2428, 2443, 2466, 2470, 2471, 2475, 2476, 2491, 2514, 2518, 2519, 2523, 2559, 2562, 2563, 2566, 2567, 2568, 2570, 2571, 2572, 2575, 2582, 2587, 2607, 2611, 2615, 2616, 2618, 2619, 2620, 2623, 2630, 2635, 2652, 2656,
											2659, 2660, 2663, 2664, 2667, 2668, 2670, 2671, 2672, 2678, 2679, 2683, 2697, 2701, 2704, 2705, 2707, 2708, 2712, 2716, 2717, 2719, 2720, 2723, 2725, 2726, 2727, 2731, 2755, 2760, 2764, 2767, 2774, 2779, 2800, 2804, 2808, 2812, 2816, 2822, 2823, 2827, 2845, 2849, 2852, 2856, 2861, 2869, 2870, 2871,
											2890, 2894, 2897, 2900, 2905, 2909, 2913, 2919, 2952, 2966, 2996, 3015, 3041, 3053, 3086, 3097, 3131, 3141};

				MatrixXd C = MatrixXd::Zero(48, 66);
				for (int i = 0; i < 768; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				MatrixXd C0 = C.leftCols(48);
				MatrixXd C1 = C.rightCols(18);
				MatrixXd C12 = C0.fullPivLu().solve(C1);
				MatrixXd RR(24, 18);
				RR << -C12.bottomRows(6), MatrixXd::Identity(18, 18);

				static const int AM_ind[] = {7, 8, 9, 10, 0, 1, 11, 2, 3, 12, 13, 14, 4, 15, 16, 17, 18, 5};
				MatrixXd AM(18, 18);
				for (int i = 0; i < 18; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				EigenSolver<MatrixXd> es(AM);
				ArrayXcd D = es.eigenvalues();
				ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(0).replicate(18, 1)).eval();

				MatrixXcd sols(2, 18);
				sols.row(0) = D.transpose(); // f^2 truth 1
				sols.row(1) = V.row(13);	 // distortion truth -0.01

				for (size_t k = 0; k < 18; ++k)
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

				return models_.size() > 0;
			} // namespace solver

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac

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
			class StitchingGravityCalib : public SolverEngine
			{
			public:
				// StitchingGravityCalib() : gravity_source(Eigen::Matrix3d::Identity()),
				// 						  gravity_destination(Eigen::Matrix3d::Identity())
				// {
				// }

				~StitchingGravityCalib()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 1;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr size_t maximumSolutions()
				{
					return 2;
				}

				static constexpr char *getName()
				{
					return "Gravity Equal Focal (1PC)";
				}

				static constexpr bool needsGravity()
				{
					return true;
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

			protected:
				Eigen::Matrix3d gravity_source;
				Eigen::Matrix3d gravity_destination;
			};

			OLGA_INLINE bool StitchingGravityCalib::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;

				// Building the coefficient matrices
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t cols = data_.cols;

				const size_t pointNumber = sample_number_; // 1

				const size_t &sampleIdx = sample_[0];

				const double
					&u1 = data_.at<double>(sampleIdx, 4), // calibrated points
					&v1 = data_.at<double>(sampleIdx, 5),
					&u2 = data_.at<double>(sampleIdx, 6),
					&v2 = data_.at<double>(sampleIdx, 7);

				if (gravity_source(0, 0) != 1)
				{
					double a0 = gravity_source(0, 0);
					double a1 = gravity_source(0, 1);
					double a2 = gravity_source(1, 0);
					double a3 = gravity_source(1, 1);
					double a4 = gravity_source(1, 2);
					double a5 = gravity_source(2, 0);
					double a6 = gravity_source(2, 1);
					double a7 = gravity_source(2, 2);

					double b0 = gravity_destination(0, 0);
					double b1 = gravity_destination(0, 1);
					double b2 = gravity_destination(1, 0);
					double b3 = gravity_destination(1, 1);
					double b4 = gravity_destination(1, 2);
					double b5 = gravity_destination(2, 0);
					double b6 = gravity_destination(2, 1);
					double b7 = gravity_destination(2, 2);

					double d1 = a0*u1 + a1*v1;
					double d2 = a4 + a2*u1 + a3*v1;
					double d3 = a7 + a5*u1 + a6*v1;

					double e1 = b0*u2 + b1*v2;
					double e2 = b4 + b2*u2 + b3*v2;
					double e3 = b7 + b5*u2 + b6*v2;


					double c0 = d2*e3 - d3*e2;
					double c1 = 2*d1*e2;
					double c2 = d2*e3 + d3*e2;
					
					double rsqr = c1*c1-4*c0*c2;
					std::complex<double> roots[2];
					roots[0] = 0.5*(-c1 + sqrt(rsqr))/c2;
					roots[1] = 0.5*(-c1 - sqrt(rsqr))/c2;

					for (size_t k = 0; k < 2; ++k)
					{
						if (roots[k].imag() > std::numeric_limits<double>::epsilon()) // cos(theta) 1-s^2 > 0
							continue;

						double r = std::clamp(roots[k].real(), -1.0, 1.0);
						double rsqr = r * r;

						Eigen::Matrix<double, 3, 3> Ry;
						Ry << (1 - rsqr), 0, 2 * r,
							0, 1 + rsqr, 0,
							-2 * r, 0, (1 - rsqr);

						Eigen::Matrix<double, 3, 3> H;
						H = gravity_destination.transpose() * Ry * gravity_source;

						Homography model;
						model.descriptor = H;
						models_.push_back(model);
					}



				}

				else
				{
					double c0 = v1 - v2;
					double c1 = 2*u1*v2;
					double c2 = v1 + v2;
					
					
					double rsqr = c1*c1-4*c0*c2;
					std::complex<double> roots[2];
					roots[0] = 0.5*(-c1 + sqrt(rsqr))/c2;
					roots[1] = 0.5*(-c1 - sqrt(rsqr))/c2;

					for (size_t k = 0; k < 2; ++k)
					{
						if (roots[k].imag() > std::numeric_limits<double>::epsilon() || roots[k].real() > 1 || roots[k].real() < -1 ) // cos(theta) 1-s^2 > 0
							continue;

						double r = std::clamp(roots[k].real(), -1.0, 1.0);
						double rsqr = r * r;

						Eigen::Matrix<double, 3, 3> Ry;
						Ry << (1 - rsqr), 0, 2 * r,
							0, 1 + rsqr, 0,
							-2 * r, 0, (1 - rsqr);

						Eigen::Matrix<double, 3, 3> H;
						H = gravity_destination.transpose() * Ry * gravity_source;

						Homography model;
						model.descriptor = H;
						models_.push_back(model);
					}
				}

				return models_.size() > 0;
			}

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac

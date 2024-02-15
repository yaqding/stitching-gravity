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
			class StitchingGeneralRadialEqual5pc : public SolverEngine
			{
			public:
				StitchingGeneralRadialEqual5pc() : gravity_source(Eigen::Matrix3d::Identity()),
												   gravity_destination(Eigen::Matrix3d::Identity())
				{
				}

				~StitchingGeneralRadialEqual5pc()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 5;
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
					return false;
				}

				static constexpr char *getName()
				{
					return "General Equal Radial (5PC)";
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

				Eigen::Matrix3d K1, K2;

			protected:
				Eigen::Matrix3d gravity_source;
				Eigen::Matrix3d gravity_destination;
			};

			OLGA_INLINE bool StitchingGeneralRadialEqual5pc::estimateModel(
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
				const size_t pointNumber = sampleSize(); // 5

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
					&v6 = data_.at<double>(sample_[2], 3),
					&u7 = data_.at<double>(sample_[3], 0),
					&v7 = data_.at<double>(sample_[3], 1),
					&u8 = data_.at<double>(sample_[3], 2),
					&v8 = data_.at<double>(sample_[3], 3),
					&u9 = data_.at<double>(sample_[4], 0),
					&v9 = data_.at<double>(sample_[4], 1),
					&u10 = data_.at<double>(sample_[4], 2),
					&v10 = data_.at<double>(sample_[4], 3);

				double r1 = u1 * u1 + v1 * v1,
					   r2 = u2 * u2 + v2 * v2,
					   r3 = u3 * u3 + v3 * v3,
					   r4 = u4 * u4 + v4 * v4,
					   r5 = u5 * u5 + v5 * v5,
					   r6 = u6 * u6 + v6 * v6,
					   r7 = u7 * u7 + v7 * v7,
					   r8 = u8 * u8 + v8 * v8,
					   r9 = u9 * u9 + v9 * v9,
					   r10 = u10 * u10 + v10 * v10;

				MatrixXd C1 = MatrixXd::Zero(9, 9);
				C1 << u2, v2, 1, 0, 0, 0, -u1 * u2, -u1 * v2, -u1,
					u4, v4, 1, 0, 0, 0, -u3 * u4, -u3 * v4, -u3,
					u6, v6, 1, 0, 0, 0, -u5 * u6, -u5 * v6, -u5,
					u8, v8, 1, 0, 0, 0, -u7 * u8, -u7 * v8, -u7,
					u10, v10, 1, 0, 0, 0, -u9 * u10, -u9 * v10, -u9,
					-v1 * u2, -v1 * v2, -v1, u1 * u2, u1 * v2, u1, 0, 0, 0,
					-v3 * u4, -v3 * v4, -v3, u3 * u4, u3 * v4, u3, 0, 0, 0,
					-v5 * u6, -v5 * v6, -v5, u5 * u6, u5 * v6, u5, 0, 0, 0,
					-v7 * u8, -v7 * v8, -v7, u7 * u8, u7 * v8, u7, 0, 0, 0;

				MatrixXd C2 = MatrixXd::Zero(9, 9);
				C2 << r1 * u2, r1 * v2, r1 + r2, 0, 0, 0, 0, 0, -r2 * u1,
					r3 * u4, r3 * v4, r3 + r4, 0, 0, 0, 0, 0, -r4 * u3,
					r5 * u6, r5 * v6, r5 + r6, 0, 0, 0, 0, 0, -r6 * u5,
					r7 * u8, r7 * v8, r7 + r8, 0, 0, 0, 0, 0, -r8 * u7,
					r9 * u10, r9 * v10, r9 + r10, 0, 0, 0, 0, 0, -r10 * u9,
					0, 0, -r2 * v1, 0, 0, r2 * u1, 0, 0, 0,
					0, 0, -r4 * v3, 0, 0, r4 * u3, 0, 0, 0,
					0, 0, -r6 * v5, 0, 0, r6 * u5, 0, 0, 0,
					0, 0, -r8 * v7, 0, 0, r8 * u7, 0, 0, 0;

				MatrixXd C3 = MatrixXd::Zero(9, 9);
				C3 << 0, 0, r2 * r1, 0, 0, 0, 0, 0, 0,
					0, 0, r4 * r3, 0, 0, 0, 0, 0, 0,
					0, 0, r6 * r5, 0, 0, 0, 0, 0, 0,
					0, 0, r8 * r7, 0, 0, 0, 0, 0, 0,
					0, 0, r10 * r9, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0;

				MatrixXd M(9, 18);
				M << C3, C2;
				M = (-C1.fullPivLu().solve(M)).eval();

				MatrixXd K(18, 18);
				K.setZero();
				K(0, 9) = 1;
				K(1, 10) = 1;
				K(2, 11) = 1;
				K(3, 12) = 1;
				K(4, 13) = 1;
				K(5, 14) = 1;
				K(6, 15) = 1;
				K(7, 16) = 1;
				K(8, 17) = 1;

				K.block<9, 18>(9, 0) = M;

				EigenSolver<MatrixXd> es(K);
				ArrayXcd D = es.eigenvalues();
				ArrayXXcd V = es.eigenvectors();

				MatrixXcd sols(1, 18);
				sols.row(0) = 1. / D.transpose();

				for (size_t k = 0; k < 18; ++k) // std::numeric_limits<double>::epsilon()
				{
					if (abs(sols(0, k).imag()) > 0.01 || std::isnan(sols(0, k).real()) || std::isinf(sols(0, k).real()) || abs(sols(0, k).real())>1.0)
						continue;

					
					Eigen::Matrix3d H;
					H << V(0, k).real(), V(1, k).real(), V(2, k).real(),
						V(3, k).real(), V(4, k).real(), V(5, k).real(),
						V(6, k).real(), V(7, k).real(), V(8, k).real();

					if (H.hasNaN())
						continue;

						std::cout << H.transpose() << std::endl;

					RadialHomography model;
					model.descriptor.block<3, 3>(0, 0) = H.transpose();

					model.descriptor(0, 3) = sols(0, k).real();
					model.descriptor(1, 3) = sols(0, k).real();
					models_.push_back(model);
				}

				if ( !(models_.size() > 0))
				{
					RadialHomography model;
					model.descriptor.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
					model.descriptor(0, 3) = 0;
					model.descriptor(1, 3) = 0;
					models_.push_back(model);
				}

				return models_.size() > 0;
			} // namespace solver

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac

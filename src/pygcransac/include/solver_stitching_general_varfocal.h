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
			class StitchingGeneralVar : public SolverEngine
			{
			public:
				StitchingGeneralVar() : gravity_source(Eigen::Matrix3d::Identity()),
										gravity_destination(Eigen::Matrix3d::Identity())
				{
				}

				~StitchingGeneralVar()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}
				
				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				static constexpr char * getName()
				{
					return "General Var Focal (3PC)";
				}

				static constexpr size_t maximumSolutions()
				{
					return 3;
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

			OLGA_INLINE bool StitchingGeneralVar::estimateModel(
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
				const size_t pointNumber = sample_number_; // 3

				// this solver is stable

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

				double d[10] = {0};

				d[0] = u1 * u3 + v1 * v3;
				d[1] = u1 * u1 + v1 * v1;
				d[2] = u3 * u3 + v3 * v3;
				d[3] = u2 * u4 + v2 * v4;
				d[4] = u2 * u2 + v2 * v2;
				d[5] = u4 * u4 + v4 * v4;
				d[6] = u1 * u5 + v1 * v5;
				d[7] = u5 * u5 + v5 * v5;
				d[8] = u2 * u6 + v2 * v6;
				d[9] = u6 * u6 + v6 * v6;

				VectorXd coeffs(16);
				coeffs[0] = -2 * d[3] + d[4] + d[5];
				coeffs[1] = 2 * d[0] - d[1] - d[2];
				coeffs[2] = -std::pow(d[3], 2) + d[4] * d[5];
				coeffs[3] = -2 * d[1] * d[3] - 2 * d[2] * d[3] + 2 * d[0] * d[4] + 2 * d[0] * d[5];
				coeffs[4] = std::pow(d[0], 2) - d[1] * d[2];
				coeffs[5] = -d[1] * std::pow(d[3], 2) - d[2] * std::pow(d[3], 2) + 2 * d[0] * d[4] * d[5];
				coeffs[6] = -2 * d[1] * d[2] * d[3] + std::pow(d[0], 2) * d[4] + std::pow(d[0], 2) * d[5];
				coeffs[7] = -d[1] * d[2] * std::pow(d[3], 2) + std::pow(d[0], 2) * d[4] * d[5];
				coeffs[8] = d[4] - 2 * d[8] + d[9];
				coeffs[9] = -d[1] + 2 * d[6] - d[7];
				coeffs[10] = -std::pow(d[8], 2) + d[4] * d[9];
				coeffs[11] = 2 * d[4] * d[6] - 2 * d[1] * d[8] - 2 * d[7] * d[8] + 2 * d[6] * d[9];
				coeffs[12] = std::pow(d[6], 2) - d[1] * d[7];
				coeffs[13] = -d[1] * std::pow(d[8], 2) - d[7] * std::pow(d[8], 2) + 2 * d[4] * d[6] * d[9];
				coeffs[14] = d[4] * std::pow(d[6], 2) - 2 * d[1] * d[7] * d[8] + std::pow(d[6], 2) * d[9];
				coeffs[15] = -d[1] * d[7] * std::pow(d[8], 2) + d[4] * std::pow(d[6], 2) * d[9];

				static const int coeffs_ind[] = {0, 8, 2, 0, 8, 10, 2, 10, 1, 9, 3, 1, 9, 11, 5, 3, 11, 13, 5, 13, 4, 12, 6, 4, 12, 14, 7, 6, 14, 15, 7, 15};

				static const int C_ind[] = {0, 3, 4, 5, 6, 7, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42};

				MatrixXd C = MatrixXd::Zero(4, 11);
				for (int i = 0; i < 32; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				MatrixXd C0 = C.leftCols(4);
				MatrixXd C1 = C.rightCols(7);
				MatrixXd C12 = C0.fullPivLu().solve(C1);
				MatrixXd RR(11, 7);
				RR << -C12.bottomRows(4), MatrixXd::Identity(7, 7);

				static const int AM_ind[] = {0, 1, 2, 3, 4, 5, 6};
				MatrixXd AM(7, 7);
				for (int i = 0; i < 7; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				EigenSolver<MatrixXd> es(AM);
				ArrayXcd D = es.eigenvalues();
				ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(6).replicate(7, 1)).eval();

				MatrixXcd sols(2, 7);
				sols.row(0) = D.transpose(); // f1^2
				sols.row(1) = V.row(5);		 // f2^2

				for (size_t k = 0; k < 3; ++k)
				{
					if (sols(0, k).imag() > std::numeric_limits<double>::epsilon() || sols(0, k).real() < 0 || sols(1, k).real() < 0)
						continue;

					double focal_1n = sqrt(sols(0, k).real());
					double focal_2n = sqrt(sols(1, k).real());
					Eigen::Vector3d u11(u1 / focal_1n, v1 / focal_1n, 1);
					Eigen::Vector3d u21(u2 / focal_2n, v2 / focal_2n, 1);

					Eigen::Vector3d u12(u3 / focal_1n, v3 / focal_1n, 1);
					Eigen::Vector3d u22(u4 / focal_2n, v4 / focal_2n, 1);

					Eigen::Vector3d u13(u5 / focal_1n, v5 / focal_1n, 1);
					Eigen::Vector3d u23(u6 / focal_2n, v6 / focal_2n, 1);

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
					K2 << focal_2n, 0, 0,
						0, focal_2n, 0,
						0, 0, 1;
					K1 << focal_1n, 0, 0,
						0, focal_1n, 0,
						0, 0, 1;
					Eigen::Matrix3d H = K2 * R * K1.inverse();

					Homography model;
					model.descriptor = Eigen::MatrixXd(3, 4);
					model.descriptor.block<3, 3>(0, 0) = H;
					model.descriptor(2, 3) = sqrt(focal_1n * focal_2n);
					models_.push_back(model);
				}

				return models_.size() > 0;
			}

		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac

#include <vector>	
#include <thread>
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"

#include "GCRANSAC.h"
#include "flann_neighborhood_graph.h"
#include "grid_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "radial_homography_estimator.h"
#include "essential_estimator.h"
#include "rigid_transformation_estimator.h"
#include "preemption_sprt.h"
#include "preemption_empty.h"

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"
#include "estimators/rigid_transformation_estimator.h"

#include "preemption/preemption_sprt.h"
#include "types.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_rigid_transformation_svd.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_p3p.h"
#include "estimators/solver_dls_pnp.h"

#include "estimators/solver_essential_matrix_one_focal_four_point.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif 

struct stat info;

template<typename _EstimatorClass>
void runTest(
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const Eigen::Matrix3d &kGravitySource_,
	const Eigen::Matrix3d &kGravityDestination_,
	Eigen::Matrix3d &kGroundTruthRotation_,
	Eigen::Vector3d &kGroundTruthTranslation_,
	const double kGroundTruthFocalLength_,
	gcransac::utils::RANSACStatistics &statistics_, 
	double &rotationError_, 
	double &translationError_,
	const double kInlierOutlierThreshold_,
	const double kSpatialWeight_,
	const double kConfidence_,
	const size_t kMaximumIterations_,
	const size_t kMinimumIterations_);

void detectFeatures(
	const std::string &kWorkspacePath_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	cv::Mat& correspondences_);

void poseError(
	const cv::Mat &R1_,
	const cv::Mat &R2_,
	const cv::Mat &t1_,
	const cv::Mat &t2_,
	double &rotationError_,
	double &translationError_);

template<typename _EstimatorClass>
void processImage(
		const std::string &kPath_,
		const std::string &kFrameSource_,
		const std::string &kFrameDestination_);

template<typename _EstimatorClass>
void processVideo(
	const std::string &kPath_,
	const size_t kImageNumber_,
	const size_t kOffset_,
	const std::string &kOutputFile_,
	const size_t kCoreNumber_);

using namespace gcransac;

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));

	std::string outputFile = "results.csv";
	size_t coreNumber = 4;
	bool removeOldResults = true;

	if (removeOldResults)
	{
		std::fstream file(outputFile);
		file.close();
	}

	// The default estimator class
	typedef estimator::EssentialMatrixEstimator<estimator::solver::EssentialOnefocal4PC, // The solver used for fitting a model to a minimal sample
		estimator::solver::EssentialMatrixBundleAdjustmentSolver> // The solver used for fitting a model to a non-minimal sample
		EstimatorClass;

	processVideo<gcransac::utils::DefaultFundamentalMatrixEstimator>(
		"/home/deep/vodata/ios_data/01/00/image_0", 1545, 10, outputFile, coreNumber);	
	processVideo<EstimatorClass>(
		"/home/deep/vodata/ios_data/01/00/image_0/", 1545, 10, outputFile, coreNumber);	
	
	return 0;
}

template<typename _EstimatorClass>
void processVideo(
	const std::string &kPath_,
	const size_t kImageNumber_,
	const size_t kOffset_,
	const std::string &kOutputFile_,
	const size_t kCoreNumber_)
{
	Eigen::Matrix3d cameraIntrinsics;
	cameraIntrinsics << 1080.098755, 0.0, 639.5, 
		0.0, 1080.098755, 359.500000,
		0.0, 0.0, 1.0;
	const double kFocalLength = 1080.098755;

	const cv::Mat kTempImage(2000, 2000, CV_64F);

	// Initializing the estimator
	std::unique_ptr<_EstimatorClass> estimator;
	if constexpr (std::is_same<gcransac::utils::DefaultFundamentalMatrixEstimator, _EstimatorClass>())
		estimator = std::unique_ptr<_EstimatorClass>(new _EstimatorClass());
	else
		estimator = std::unique_ptr<_EstimatorClass>(new _EstimatorClass(cameraIntrinsics, cameraIntrinsics));

	std::mutex savingGuard;

#pragma omp parallel for num_threads(kCoreNumber_)
	for (size_t frame = 0; frame < kImageNumber_ - kOffset_; ++frame)
	{
		const size_t &kFrameSource = frame,
			kFrameDestination = frame + kOffset_;
		
		std::string paddingSource = "";
		for (size_t padding = 0; padding < 6 - std::to_string(kFrameSource).size(); ++padding)
			paddingSource += "0";
			
		std::string paddingDestination = "";
		for (size_t padding = 0; padding < 6 - std::to_string(kFrameDestination).size(); ++padding)
			paddingDestination += "0";

		printf("Processing frames %d and %d\n", kFrameSource, kFrameDestination);

		// Read gravity
		const std::string kSourceGravityPath = kPath_ + "frame_" + paddingSource + std::to_string(kFrameSource) + "_gravity.txt",
			kDestinationGravityPath = kPath_ + "frame_" + paddingDestination + std::to_string(kFrameDestination) + "_gravity.txt";

		Eigen::Matrix3d sourceGravity, destinationGravity;

		gcransac::utils::loadMatrix<double, 3, 3>(kSourceGravityPath, sourceGravity);
		gcransac::utils::loadMatrix<double, 3, 3>(kDestinationGravityPath, destinationGravity);

		// Read pose
		const std::string kSourcePosePath = kPath_ + "frame_" + paddingSource + std::to_string(kFrameSource) + "_pose.txt",
			kDestinationPosePath = kPath_ + "frame_" + paddingDestination + std::to_string(kFrameDestination) + "_pose.txt";
		Eigen::Matrix<double, 3, 4> poseSource, poseDestination;
		gcransac::utils::loadMatrix<double, 3, 4>(kSourcePosePath, poseSource);
		gcransac::utils::loadMatrix<double, 3, 4>(kDestinationPosePath, poseDestination);

		Eigen::Matrix3d rotationSource = poseSource.block<3, 3>(0, 0),
			rotationDestination = poseDestination.block<3, 3>(0, 0);

		Eigen::Vector3d translationSource = poseSource.block<1, 3>(3, 0),
			translationDestination = poseDestination.block<1, 3>(3, 0);

		// Eigen::Matrix3d relativeRotation =
		//  	rotationDestination * rotationSource.transpose();
		// Eigen::Vector3d relativeTranslation =
		// 	translationDestination - rotationSource.transpose() * translationSource;

		Eigen::Matrix3d relativeRotation = rotationSource;
		Eigen::Vector3d relativeTranslation = translationSource;

		// Read correspondences
		const std::string kCorrespondencePath = 
			kPath_ + "frame_" + std::to_string(kFrameSource) + "_" + std::to_string(kFrameDestination) + "_corrs.txt";
			
		cv::Mat correspondences;		
		gcransac::utils::loadPointsFromFile<8, 1, 0>(
			correspondences,
			kCorrespondencePath.c_str()),

		printf("Loaded matches = %d\n", correspondences.rows);

		// We don't need the SIFT parameters so remove them
		correspondences = correspondences(cv::Rect(0, 0, 4, correspondences.rows));

		// Normalizing correspondences
		cv::Mat normalizedCorrespondences(correspondences.size(), correspondences.type());
		utils::normalizeCorrespondences(
				correspondences,
				cameraIntrinsics,
				cameraIntrinsics,
				normalizedCorrespondences);
				
		// Run tests
		utils::RANSACStatistics statistics;
		double rotationError, translationError;

		runTest<_EstimatorClass>(
				correspondences, normalizedCorrespondences,
				kTempImage, kTempImage,
				cameraIntrinsics, cameraIntrinsics, 
				sourceGravity, destinationGravity,
				relativeRotation, relativeTranslation, kFocalLength,
				statistics, rotationError, translationError,
				0.75, 0.0, 0.999, 10000, 20);

		// Print the statistics
		printf("Elapsed time = %f secs\n", statistics.processing_time);
		printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
		printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
		printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
		printf("Number of iterations = %d\n", static_cast<int>(statistics.iteration_number));
		printf("Rotation error = %f degrees\n", rotationError);
		printf("Translation error = %f degrees\n\n", translationError);

		savingGuard.lock();
		std::ofstream file(kOutputFile_, std::fstream::app);
		file << kFrameSource << ";" 
			<< kFrameDestination << ";"
			<< estimator->getMinimalSolver()->getName() << ";"
			<< estimator->getNonMinimalSolver()->getName() << ";"
			<< statistics.processing_time << ";"
			<< statistics.inliers.size() << ";"
			<< statistics.iteration_number << ";"
			<< rotationError << ";"
			<< translationError << "\n";
		file.close();
		savingGuard.unlock();
	}
}


template<typename _EstimatorClass>
void runTest(
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const Eigen::Matrix3d &kGravitySource_,
	const Eigen::Matrix3d &kGravityDestination_,
	Eigen::Matrix3d &kGroundTruthRotation_,
	Eigen::Vector3d &kGroundTruthTranslation_,
	const double kGroundTruthFocalLength_,	
	utils::RANSACStatistics &statistics_, 
	double &rotationError_, 
	double &translationError_,
	const double kInlierOutlierThreshold_,
	const double kSpatialWeight_,
	const double kConfidence_,
	const size_t kMaximumIterations_,
	const size_t kMinimumIterations_)
{	
	// Initializing the estimator
	std::unique_ptr<_EstimatorClass> estimator;
	if constexpr (std::is_same<gcransac::utils::DefaultFundamentalMatrixEstimator, _EstimatorClass>())
		estimator = std::unique_ptr<_EstimatorClass>(new _EstimatorClass());
	else
		estimator = std::unique_ptr<_EstimatorClass>(new _EstimatorClass(kIntrinsicsSource_, kIntrinsicsDestination_));

	// Setting the gravity if the solver needs it
	if (estimator->getMinimalSolver()->needsGravity())
		estimator->getMinimalSolver()->setGravity(
			kGravitySource_,
			kGravityDestination_);
			
	if (estimator->getNonMinimalSolver()->needsGravity())
		estimator->getNonMinimalSolver()->setGravity(
			kGravitySource_,
			kGravityDestination_);

	// Normalize the threshold by the average of the focal lengths
	const double kNormalizedThreshold =
		kInlierOutlierThreshold_ / ((kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) +
			kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1)) / 4.0);

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph<4> neighborhood(&kCorrespondences_,
		{ kSourceImage_.cols / static_cast<double>(4),
			kSourceImage_.rows / static_cast<double>(4),
			kDestinationImage_.cols / static_cast<double>(4),
			kDestinationImage_.rows / static_cast<double>(4) },
		4);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Apply Graph-cut RANSAC
	std::vector<int> inliers;
	EssentialMatrix model;

	// Initializing SPRT test
	preemption::EmptyPreemptiveVerfication<_EstimatorClass> preemptiveVerification;

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProsacSampler mainSampler(&kCorrespondences_, _EstimatorClass::sampleSize());  
	sampler::UniformSampler localOptimizationSampler(&kCorrespondences_); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!mainSampler.isInitialized() ||
		!localOptimizationSampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	GCRANSAC<_EstimatorClass,
		neighborhood::GridNeighborhoodGraph<4>,
		MSACScoringFunction<_EstimatorClass>,
		preemption::EmptyPreemptiveVerfication<_EstimatorClass>> gcransac;
	gcransac.settings.threshold = kNormalizedThreshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = kSpatialWeight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = kConfidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = kMaximumIterations_; // The maximum number of iterations
	gcransac.settings.min_iteration_number = kMinimumIterations_; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 4; // The radius of the neighborhood ball

	// Start GC-RANSAC
	gcransac.run(kNormalizedCorrespondences_,
		*estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhood,
		model,
		preemptiveVerification);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	double rotationError = std::numeric_limits<double>::max(),
		translationError = std::numeric_limits<double>::max();

	if (statistics.inliers.size() <= _EstimatorClass::sampleSize())
		model.descriptor = Eigen::Matrix3d::Identity();

	// Calculate the pose error
	Eigen::Matrix3d essentialMatrix = model.descriptor.block<3, 3>(0, 0);
	cv::Mat cv_EssentialMatrix(3, 3, CV_64F, essentialMatrix.data());
	cv_EssentialMatrix = cv_EssentialMatrix.t();
	cv::Mat cv_Rotation, cv_Translation;

	const int &kPointNumber = kNormalizedCorrespondences_.rows;
	std::vector<uchar> inlierMask(kPointNumber, 0);
	for (const auto &inlierIdx : statistics.inliers)
		inlierMask[inlierIdx] = 1;

	cv::recoverPose(cv_EssentialMatrix, 
		kNormalizedCorrespondences_(cv::Rect(0, 0, 2, kPointNumber)),
		kNormalizedCorrespondences_(cv::Rect(2, 0, 2, kPointNumber)),
		cv::Mat::eye(3, 3, CV_64F),
		cv_Rotation,
		cv_Translation,
		inlierMask);

	cv::Mat cv_GroundTruthRotation(3, 3, CV_64F, kGroundTruthRotation_.data());
	cv_GroundTruthRotation = cv_GroundTruthRotation.t();
	cv::Mat cv_GroundTruthTranslation(1, 3, CV_64F, kGroundTruthTranslation_.data());
	cv_GroundTruthTranslation = cv_GroundTruthTranslation.t();

	poseError(
		cv_Rotation,
		cv_GroundTruthRotation,
		cv_Translation,
		cv_GroundTruthTranslation,
		rotationError_,
		translationError_);

	statistics_ = statistics;
}

void poseError(
	const cv::Mat &R1_,
	const cv::Mat &R2_,
	const cv::Mat &t1_,
	const cv::Mat &t2_,
	double &rotationError_,
	double &translationError_)
{
	// Calculate angle between provided rotations
	cv::Mat R2R1 = R2_ * R1_.t();
	const double cos_angle =
		std::max(std::min(1.0, 0.5 * (cv::trace(R2R1).val[0] - 1.0)), -1.0);
	rotationError_ = 180.0 / M_PI * std::acos(cos_angle);
	
	// calculate angle between provided translations
	double translationError1 = t2_.dot(t1_) / cv::norm(t2_) / cv::norm(t1_);
	translationError1 = MAX(-1.0, MIN(1.0, translationError1));
	translationError1 = acos(translationError1) * 180 / M_PI;
	
	double translationError2 = t2_.dot(-t1_) / cv::norm(t2_) / cv::norm(t1_);
	translationError2 = MAX(-1.0, MIN(1.0, translationError2));
	translationError2 = acos(translationError2) * 180 / M_PI;

	translationError_ = MIN(translationError1, translationError2);
}

void detectFeatures(
	const std::string &kWorkspacePath_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	cv::Mat& correspondences_)
{
	if (gcransac::utils::loadPointsFromFile(correspondences_,
		kWorkspacePath_.c_str()))
	{
		printf("Match number = %d\n", correspondences_.rows);
		return;
	}

	cv::Mat descriptors[2];
	std::vector<cv::KeyPoint> keypoints[2];

	cv::Ptr<cv::SiftFeatureDetector> sift = cv::SIFT::create(10000, 3, 0.0, 10000.);
	sift->detect(kSourceImage_, keypoints[0]);
	sift->compute(kSourceImage_, keypoints[0], descriptors[0]);

	sift->detect(kDestinationImage_, keypoints[1]);
	sift->compute(kDestinationImage_, keypoints[1], descriptors[1]);
	 
	// Normalize the descriptor vectors to get RootSIFT
	for (auto& descriptor : descriptors)
	{
		for (size_t row = 0; row < descriptor.rows; ++row)
		{
			descriptor.row(row) *= 1.0 / cv::norm(descriptor.row(row), cv::NORM_L1);
			for (size_t col = 0; col < descriptor.cols; ++col)
				descriptor.at<float>(row, col) = std::sqrt(descriptor.at<float>(row, col));
		}
	}

	// Do brute-force matching from the source to the destination image
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(descriptors[0], descriptors[1], matches, 2);

	// Do brute-force matching from the destination to the source image
	cv::Ptr<cv::DescriptorMatcher> matcher_opposite = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
	std::vector<std::vector<cv::DMatch>> matches_opposite;
	matcher_opposite->knnMatch(descriptors[1], descriptors[0], matches_opposite, 1);

	std::vector<std::pair<double, const std::vector<cv::DMatch>*>> good_matches;

	// Do mutual nearest neighbor search
	for (size_t i = 0; i < matches.size(); ++i)
		if ((matches[i][0].distance < 0.9 * matches[i][1].distance) &&
			(matches[i][0].queryIdx == matches_opposite[matches[i][0].trainIdx][0].trainIdx)) // We increased threshold for mutual snn check
			good_matches.emplace_back(std::make_pair(matches[i][0].distance / matches[i][1].distance, &matches[i]));

	// Sort the correspondences according to their distance.
	// This is done for using PROSAC sampling
	std::sort(good_matches.begin(), good_matches.end());

	// Create the container for the correspondences
	correspondences_.create(good_matches.size(), 4, CV_64F);
	double* correspondences_ptr = reinterpret_cast<double*>(correspondences_.data);

	// Fill the container by the selected matched
	for (const auto& match_ptr : good_matches)
	{
		const std::vector<cv::DMatch>& match = *match_ptr.second;

		*(correspondences_ptr++) = keypoints[0][match[0].queryIdx].pt.x;
		*(correspondences_ptr++) = keypoints[0][match[0].queryIdx].pt.y;
		*(correspondences_ptr++) = keypoints[1][match[0].trainIdx].pt.x;
		*(correspondences_ptr++) = keypoints[1][match[0].trainIdx].pt.y;
	}

	gcransac::utils::savePointsToFile(correspondences_, 
		kWorkspacePath_.c_str());
	printf("Match number = %d\n", correspondences_.rows);
}
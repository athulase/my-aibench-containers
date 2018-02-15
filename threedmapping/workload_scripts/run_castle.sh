#!/bin/bash

source ~/.bashrc

#delete output so new run doesn't use existing features/match data
LOG_DIR=$THREED_LOG_DIR
mkdir -p ${LOG_DIR}/output
mkdir -p ${CASTLE_PATH}/features

start_time=`date +%s`

#start reconstruction
/theia-build/bin/build_reconstruction --output_matches_file=${LOG_DIR}/output/${HOSTNAME}-matches.bat \
--images=${CASTLE_PATH}/*.png \
--matching_working_directory=${CASTLE_PATH}/features \
--output_reconstruction=${LOG_DIR}/output \
--num_threads=${THEIA_THREADS} \
--descriptor=SIFT \
--feature_density=NORMAL \
--match_out_of_core=true \
--matching_max_num_images_in_cache=64 \
--matching_strategy=CASCADE_HASHING \
--lowes_ratio=0.7 \
--min_num_inliers_for_valid_match=30 \
--max_sampson_error_for_verified_match=4.0 \
--bundle_adjust_two_view_geometry=true \
--keep_only_symmetric_matches=true \
--reconstruction_estimator=GLOBAL \
--min_track_length=3 \
--max_track_length=50 \
--reconstruct_largest_connected_component=true \
--shared_calibration=true \
--only_calibrated_views=false \
--global_position_estimator=LEAST_UNSQUARED_DEVIATION \
--global_rotation_estimator=ROBUST_L1L2 \
--post_rotation_filtering_degrees=20.0 \
--refine_relative_translations_after_rotation_estimation=true \
--extract_maximal_rigid_subgraph=false \
--filter_relative_translations_with_1dsfm=true \
--position_estimation_min_num_tracks_per_view=0 \
--position_estimation_robust_loss_width=0.01 \
--num_retriangulation_iterations=1 \
--refine_camera_positions_and_points_after_position_estimation=false \
--absolute_pose_reprojection_error_threshold=4 \
--partial_bundle_adjustment_num_views=20 \
--full_bundle_adjustment_growth_percent=5 \
--min_num_absolute_pose_inliers=30 \
--bundle_adjustment_robust_loss_function=HUBER \
--bundle_adjustment_robust_loss_width=10.0 \
--intrinsics_to_optimize=NONE \
--max_reprojection_error_pixels=4.0 \
--min_triangulation_angle_degrees=4.0 \
--triangulation_reprojection_error_pixels=10.0 \
--bundle_adjust_tracks=true > ${LOG_DIR}/output/${HOSTNAME}-output-br.log 2>&1

#convert output mesh to viewable .ply file
/theia-build/bin/write_reconstruction_ply_file --reconstruction=${LOG_DIR}/output-0 --ply_file=${LOG_DIR}/output/output-0.ply > ${LOG_DIR}/output/${HOSTNAME}-output-wr.log 2>&1

# Save the throughput
end_time=`date +%s`
timetaken=$((end_time-start_time))
file_count=`ls $CASTLE_PATH/*.png | wc -l`
result=$(echo ${file_count}/${timetaken}|bc -l)

echo "$result ImagesPerSecond" >> /relevant_metrics
echo "1.0 AverageConfidence" >> /relevant_metrics
cp /relevant_metrics $LOG_DIR/relevant_metrics

import argparse
import json
import os.path
import pytz
import shutil
import sys
import pandas as pd
from adwsdk.adw_client import AdwClient
from concurrent.futures import ThreadPoolExecutor
from datafilter.cli.download_utils import download_subclip, download_one_clip
from datafilter.sdk.clip import ClipInfoSDK
from datafilter.sdk.rawdatareader import *
from datafilter.sdk.rawdatareader import _parse_rtk_json, _parse_can_json, _parse_imu_json, parse_can_from_pb, \
    parse_imu_from_pb, parse_gnss_from_pb
from datafilter.sdk.rawdatareader import get_calibration
from datafilter.src.log.logger import logger
from da_data_sdk.clip import ClipDownloader
from datetime import datetime

os.environ['ADW_USER']='datafilter'
os.environ['ADW_PASS']='g9N4vEYiVL'
os.environ['ADW_USER_STG']='zhiguang.zhang'
os.environ['ADW_PASS_STG']='RX3PAIDVXH'

SENSOR_TOPICS = dict([
    ('FW', '/camera/front/main'),
    ('FN', '/camera/front/narrow'),
    ('FL', '/camera/side/front/left'),
    ('FR', '/camera/side/front/right'),

    ('RL', '/camera/side/rear/left'),
    ('RR', '/camera/side/rear/right'),
    ('RN', '/camera/rear'),

    ('SLW', '/camera/surrouding/left'),
    ('SRW', '/camera/surrouding/right'),
    ('SFW', '/camera/surrouding/front'),
    ('SRCW', '/camera/surrouding/rear'),
    ('FALCON', '/lidar/falcon'),
    ('PANDAR128', '/lidar/pandar128'),
    ('QT64_Left', '/lidar/hesai/qt64/left'),
    ('QT64_Right', '/lidar/hesai/qt64/right'),
    ('RTK', '/rtk1'),
    ('CAN', '/can/chassis'),

    ('CIL', '/psap/camera/inside/left'),
    ('CIR', '/psap/camera/inside/right'),
    ('COL1', '/psap/camera/outside/left1'),
    ('COL2', '/psap/camera/outside/left2'),
    ('COR1', '/psap/camera/outside/right1'),
    ('COR2', '/psap/camera/outside/right2'),
    ('COM', '/psap/camera/outside/middle'),
    ('QT128IL', '/psap/lidar/qt128/inside/left'),
    ('QT128IR', '/psap/lidar/qt128/inside/right'),
    ('QT128OL', '/psap/lidar/qt128/outside/left'),
    ('QT128OR', '/psap/lidar/qt128/outside/right'),
    ('IMU', '/imu/raw'),
    ('GNSS', '/rtk/raw')
])

DLB_TOPICS = dict([
    ('CAN', '/common/vehicle_in/vehicle_10ms'),
    ('IMU', '/common/sensors/imu'),
    ('GNSS', '/common/sensors/gnss_result')
])

CAMERAS = ['FW',
           'FN',
           'FL',
           'FR',
           'RL',
           'RR',
           'RN',
           'SLW',
           'SRW',
           'SFW',
           'SRCW',
           'CIL',
           'CIR',
           'COL1',
           'COL2',
           'COR1',
           'COR2',
           'COM', ]

HESAI_LIDARS = ['PANDAR128',
                'QT64_Left',
                'QT64_Right',
                'QT128IL',
                'QT128IR',
                'QT128OL',
                'QT128OR',
                ]


def _get_topic_file(meta_info, topic_name):
    filename = None
    if topic_name not in meta_info['topics']:
        # logger.error('not find topic_name:{} in in meta', topic_name)
        print('not find topic_name in in meta: ' + topic_name)
        return filename

    filenames = meta_info['topics'][topic_name]
    if "file_name" in filenames:
        return filenames["file_name"]
    elif len(filenames) == 0:
        logger.error('Empty file list in topic: {}', topic_name)
        return None
    elif len(filenames) >= 1:
        filename = filenames[0]
    return filename

def patch_raw_data(clip_path):
    from datafilter.sdk.download_utils import ClipPatcher
    patch_topic_lookup = {'/lidar/pandar128': '/parse/pandar128/timeregress',
        '/camera/front/main': '/parse/FW/timeregress',
        '/camera/front/narrow': '/parse/FN/timeregress',
        '/camera/side/front/left': '/parse/FL/timeregress',
        '/camera/side/front/right': '/parse/FR/timeregress',
        '/camera/side/rear/left': '/parse/RL/timeregress',
        '/camera/side/rear/right': '/parse/RR/timeregress',
        '/camera/rear': '/parse/RN/timeregress',

        '/camera/surrouding/left': '/parse/SLW/timeregress',
        '/camera/surrouding/right': '/parse/SRW/timeregress',
        '/camera/surrouding/front': '/parse/SFW/timeregress',
        '/camera/surrouding/rear': '/parse/SRCW/timeregress',
    }
    meta_file = os.path.join(clip_path, 'meta.json')
    if not os.path.exists(meta_file):
        logger.info('clip path have no meta.json, skip patching.')
        return
    
    clip_meta = json.load(open(meta_file, 'r', encoding='utf-8'))
    clip_id_list = clip_meta.get('clip_id')
    namespace = clip_meta.get('namespace')
    if clip_id_list is None:
        logger.info('meta.json in clip_path: {} have no clip_id', meta_file)
        return
    clip_id = clip_id_list[0]
    if namespace is None:
        logger.info('meta.json in clip_path: {} have no namespace', meta_file)
        return
    patcher = ClipPatcher(clip_path, patch_cam=True)
    for topic, filenames in clip_meta.get('topics', {}).items():
        if topic in patch_topic_lookup:
            if len(filenames) == 0:
                continue
            filename = filenames[0]
            if not filename.endswith('.dat') and not filename.endswith('.h264') and not filename.endswith('.h265'):
                continue
            logger.debug('clip_path: {}, filename:{}', clip_path, filename)
            filepath = os.path.join(clip_path, filename)
            if not os.path.exists(filepath):
                logger.info('Skip patching, since filepath: {} not exist.', filepath)
                continue
            logger.info('patching clip_id: {}, topic: {}, filepath: {}', clip_id, topic, filepath)
            patcher.patch_one(topic, filepath, clip_id, namespace)

# lidar file
def parse_falcon_file(falcon_data_path, out_folder):
    reader = FalconReader(falcon_data_path)
    while True:
        frame = reader.get_frame_with_info()
        if frame is None:
            break
        pcd, frameinfo = frame
        frame_starttime = frameinfo['frame_starttime']
        frame_idx = frameinfo['frame_idx']
        packet_percent = frameinfo['packet_percent']
        if packet_percent < 0.10:
            logger.warning(
                'frame_idx: {}, frame_starttime: {}, pcd.shape: {}, frame_info: {}, packet percent lower then 0.10. ignore',
                frame_idx, frame_starttime, pcd.shape, frameinfo)
            continue
        logger.debug('frame_idx: {}, frame_starttime: {}, pcd.shape: {}',
                     frame_idx, frame_starttime, pcd.shape)

        pcd[:, 4] -= frame_starttime  # relative timestamp
        save_pcd(pcd, os.path.join(out_folder, '{:.6f}.pcd'.format(frame_starttime)), 'pandar128_binary')


def parse_hesai_lidar_point_cloud(clipfolder, sensor_name, out_folder):
    metafile = os.path.join(clipfolder, 'meta.json')
    with open(metafile, 'r') as f:
        meta_info = json.load(f)

    falcon_file_name = _get_topic_file(meta_info, SENSOR_TOPICS[sensor_name])
    if falcon_file_name is None:
        print("==========Not exist lidar data!==========")
        return
    lidar_file = os.path.join(clipfolder, falcon_file_name)
    calib_file = get_calibration_file(meta_info, clipfolder)

    os.makedirs(out_folder, exist_ok=True)
    if 'PANDAR128' in sensor_name:
        reader = Pandar128ReaderV2(lidar_file, calib_file)
    elif 'QT128' in sensor_name:
        reader = PandarQt128Reader(lidar_file, calib_file, sensor_name)
    elif 'QT64' in sensor_name:
        reader = PandarQtReader(lidar_file, calib_file, sensor_name)
    else:
        print('===================== Can not recognize the lidar {} ======================'.format(sensor_name))
        return

    while True:
        # starttime = time.time()
        frame = reader.get_next_frame()
        if frame is None:
            break
        frame_idx, frame_starttime, pcd = frame
        # usedtime = int((time.time() - starttime) * 1000)
        logger.debug(
            'frame_idx:{}, frame_starttime:{}, points:{}', frame_idx, frame_starttime, len(pcd))
        pcd[:, 4] -= frame_starttime  # relative timestamp
        save_pcd(pcd, os.path.join(out_folder, '{:.6f}.pcd'.format(frame_starttime)), 'pandar128_binary')


def parse_video(camera_videos, out_folder, num_threads=4, usegpu=0):
    print('=========================================================================')
    usegpu = int(usegpu)
    video_infos = []
    for camera_name, file_path, timestamp_file_path in camera_videos:
        video_infos.append({'camname': camera_name, 'videofile': file_path, 'timestampfile': timestamp_file_path})
    if usegpu:
        video_reader = GPUVideoReader(video_infos)
    else:
        video_reader = VideoReader(video_infos)

    video_reader.start()

    camera_infos = {}
    camera_folders = {}
    camera_names = []
    for camera_name, _, _ in camera_videos:
        camera_infos[camera_name] = video_reader.get_timestamps(camera_name)
        camera_images_folder = os.path.join(out_folder, camera_name)
        os.makedirs(camera_images_folder, exist_ok=True)
        camera_folders[camera_name] = camera_images_folder
        camera_names.append(camera_name)

    sync_frames = get_sync_frames(camera_names, camera_infos, camera_names[0])

    # multiple thread save jpeg
    threadpool = ThreadPoolExecutor(max_workers=num_threads)

    processing = []
    for i, sync_frame in enumerate(sync_frames):

        logger.debug('count {}/{}', i, len(sync_frames))
        for camera_name, timestamp in zip(camera_names, sync_frame):
            timestamp, frame = video_reader.get_frame(camera_name, timestamp)
            # logger.debug('getframe {}, {}, {}, {}, {} ', camera_name, timestamp, frame.shape, frame.dtype,
            #              time.time() - starttime)

            filename = os.path.join(camera_folders[camera_name], '{}.jpg'.format(timestamp))
            # jpgwrite(filename, frame)
            processing.append(threadpool.submit(jpgwrite, filename, frame))
            if len(processing) > 5:
                for future in processing:
                    res = future.result()
                processing = []

    # threadpool.shutdown()
    video_reader.stop()
    video_reader = None


def save_into_csv(parsed_data: pd.DataFrame, output_csv_filepath):
    output_csv_path = Path(output_csv_filepath)
    if not os.path.isdir(output_csv_path.parent):
        os.makedirs(output_csv_path.parent, exist_ok=True)
    if os.path.exists(output_csv_path):
        logger.info(f'output path exists, overwriting: {output_csv_path}')
    parsed_data.to_csv(output_csv_path, index=False)
    logger.info(f'{len(parsed_data)} records parsed and saved into output file: {output_csv_path}')


def parse_global_localization_from_pb(orgin_file, output_csv_path):
    from datafilter.sdk.rawdatareader import MazuMessagePbReader
    from datafilter.sdk.rawdatareader import pb_parser_manager
    if pb_parser_manager.PB_MODULE_VERSION is not None:
        version_key = pb_parser_manager.PB_MODULE_VERSION
    else:
        version_key = 'default'
    logger.info('PB_MODULE_VERSION: {}, version_key: {}', pb_parser_manager.PB_MODULE_VERSION, version_key)
    aa = MazuMessagePbReader(orgin_file, 'common/localization/global_localization', version_key)
    all_data = []
    while True:
        try:
            frame_idx, frame_info, data = aa.get_next_frame()
            utc_timestamp = data.timestamp
            ptp_timestamp = data.timestamp_ptp_ns / 1e9
            status = data.status
            # 3 DoFs attitude under local ENU frame, expressed using Euler angle. [radian]
            euler_angle_roll = data.euler_angle.roll
            euler_angle_pitch = data.euler_angle.pitch
            euler_angle_yaw = data.euler_angle.yaw
            # Longitude [degree], latitude [degree], and altitude [m].
            wgs84_position_x = data.wgs84_position.x
            wgs84_position_y = data.wgs84_position.y
            wgs84_position_z = data.wgs84_position.z
            velocity_x = data.velocity.x
            velocity_y = data.velocity.y
            velocity_z = data.velocity.z
            quaternion_x = data.quaternion.x
            quaternion_y = data.quaternion.y
            quaternion_z = data.quaternion.z
            quaternion_w = data.quaternion.w
            ## Ego motion of the car
            linear_acc_x = data.linear_acc.x
            linear_acc_y = data.linear_acc.y
            linear_acc_z = data.linear_acc.z
            angular_vel_x = data.angular_vel.x
            angular_vel_y = data.angular_vel.y
            angular_vel_z = data.angular_vel.z
            # // Global position under a cartesian coordinates system defined at the first frame.
            cartesian_position_x = data.cartesian_position.x
            cartesian_position_y = data.cartesian_position.y
            cartesian_position_z = data.cartesian_position.z
            # print('utc_timestamp: ', utc_timestamp, ptp_timestamp, euler_angle_roll, euler_angle_pitch, euler_angle_yaw,
            #                 position_x, position_y, position_z, velocity_x, velocity_y, velocity_z,
            #                 quaternion_x, quaternion_y, quaternion_z, quaternion_w)
            one_frame_data = [utc_timestamp, ptp_timestamp, status, \
                              euler_angle_roll, euler_angle_pitch, euler_angle_yaw, \
                              wgs84_position_x, wgs84_position_y, wgs84_position_z, \
                              velocity_x, velocity_y, velocity_z, \
                              quaternion_x, quaternion_y, quaternion_z, quaternion_w, \
                              linear_acc_x, linear_acc_y, linear_acc_z, \
                              angular_vel_x, angular_vel_y, angular_vel_z, \
                              cartesian_position_x, cartesian_position_y, cartesian_position_z]
            all_data.append(one_frame_data)
            # import ipdb; ipdb.set_trace()
        except:
            break
    columns_names = ["utc_timestamp", "ptp_timestamp", "status", \
                     "euler_angle_roll", "euler_angle_pitch", "euler_angle_yaw", \
                     "wgs84_position_x", "wgs84_position_y", "wgs84_position_z", \
                     "velocity_x", "velocity_y", "velocity_z", \
                     "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w", \
                     "linear_acc_x", "linear_acc_y", "linear_acc_z", \
                     "angular_vel_x", "angular_vel_y", "angular_vel_z", \
                     "cartesian_position_x", "cartesian_position_y", "cartesian_position_z"]
    data = pd.DataFrame(all_data, columns=columns_names)
    save_into_csv(data, output_csv_path)


def get_calibration_file(meta_info, output_dir):
    car_vin = meta_info['car_id']
    start_datetime = datetime.strptime(meta_info['start_time'], '%Y%m%dT%H%M%S')
    timezone = pytz.timezone(meta_info.get('timezone', 'Asia/Shanghai'))
    clip_start_time = int(timezone.localize(start_datetime).timestamp())

    calib_file = get_calibration(car_vin, clip_start_time, cached_folder=output_dir)
    return calib_file


def my_parse_video_time(meta_file_path, video_file_path, output_timestamp_file_path_prefix):
    timestamp_file = video_file_path + '.timestamps'
    output_timestamp_file_path = output_timestamp_file_path_prefix + '.origin_timestamps'
    if os.path.exists(timestamp_file) and output_timestamp_file_path != timestamp_file:
        shutil.copy(timestamp_file, output_timestamp_file_path)
    else:
        parse_video_time(video_file_path, output_timestamp_file_path)

    fixed_timestamp_file = output_timestamp_file_path_prefix + '.timestamps'
    fix_video_timestamp(meta_file_path, output_timestamp_file_path, fixed_timestamp_file)


def parse_args():
    usage = """python3 download_and_parse_clip.py --clip-id 4c17c407-904e-4a76-94fc-f4a7a02aa065 --output-dir ./
     --sensors FW/FN/FL/FR/FALCON/RTK/CAN"""

    parser = argparse.ArgumentParser(description='parse data.', usage=usage)
    parser.add_argument('--clip-id', dest='clip_id', help='clip id or DLB uuid', required=False)
    parser.add_argument('--output-dir', dest='output_dir', help='', required=True)
    parser.add_argument('--sensors', dest='sensors',
                        help="Optional, the sensor names to be parsed, separate sensor name by '/',"
                             " default: all existing sensors", required=False)
    parser.add_argument('--save-lidar-frames', dest='save_lidar_frames', action='store_true',
                        help='Optional, store true, whether to save lidar frame as *.pcd', required=False)
    parser.add_argument('--save-video-frames', dest='save_video_frames', action='store_true',
                        help='Optional, store true, whether to save video frame as images', required=False)
    parser.add_argument('--start-time', dest='start_time', type=float,
                        help='Optional, start time (in second) of the data segment to download and parse',
                        required=False)
    parser.add_argument('--end-time', dest='end_time', type=float,
                        help='Optional, end time (in second) of the data segment to download and parse',
                        required=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)

    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)

    return parsed


def download_and_parse_clip(output_dir, clip_id=None, sensors=None, save_lidar_frames=False, save_video_frames=False,
                            start_time=None, end_time=None):
    if sensors is None:
        sensors = list(SENSOR_TOPICS.keys())
        sensors += list(DLB_TOPICS.keys())
        topics = '*'
    else:
        sensors = sensors.split('/')
        topics = [SENSOR_TOPICS[sensor] for sensor in sensors if sensor in SENSOR_TOPICS]
        topics = ','.join(topics)
    
    clip_path = os.path.join(output_dir, clip_id)

    calibration_file_path = None
    if clip_id is not None:
        clip_downloader = ClipDownloader(clip_id)
        is_dlb_data = '_' not in clip_id  # naive method to distinguish clip id and uuid
        clip_downloader.add_topics(topics)
        if is_dlb_data:
            clip_downloader.download(clip_path)
        else:
            if start_time is not None and end_time is not None:
                clip_downloader.download(clip_path, start_time, end_time)
            else:
                clip_downloader.download(clip_path)
        
        da_meta_file = os.path.join(clip_path, 'da_data_meta.json')
        meta_file = os.path.join(clip_path, 'meta.json')
        if os.path.exists(da_meta_file) :
            os.rename(da_meta_file, meta_file)
            logger.info('rename da_meta_json to meta_json')

        with open(meta_file, 'r') as f:
            meta_info = json.load(f)
        if not is_dlb_data:
            collection_type = meta_info['nt_version'].replace('.0', '')
            car_type = meta_info['model_type']
            patch_raw_data(clip_path)

        clip_downloader.get_calibration(clip_path + "/" + clip_id +'_calibration.json')
    else:
        meta_file = os.path.join(clip_path, 'meta.json')
        with open(meta_file, 'r') as f:
            meta_info = json.load(f)
        is_dlb_data = 'uuid' in meta_info
        if not is_dlb_data:
            collection_type = meta_info['nt_version'].replace('.0', '')
            car_type = meta_info['model_type']
            patch_raw_data(clip_path)

    if 'namespace' in meta_info:
        is_roadtest = meta_info['namespace'] == 'roadtest'

    output_path = os.path.join(clip_path, 'output')
    os.makedirs(output_path, exist_ok=True)

    camera_videos = []
    for sensor in sensors:
        file_name = None
        if sensor in SENSOR_TOPICS:
            file_name = _get_topic_file(meta_info, SENSOR_TOPICS[sensor])
        if (is_dlb_data or is_roadtest) and sensor in DLB_TOPICS:
            file_name = _get_topic_file(meta_info, DLB_TOPICS[sensor])
        if file_name is None:
            logger.error('Do not find the raw data file of sensor: {}'.format(sensor))
            continue

        file_path = os.path.join(clip_path, file_name)
        if not os.path.exists(file_path):
            continue

        # parse CAN
        if sensor == 'CAN':
            can_path = os.path.join(output_path, 'can')
            os.makedirs(can_path, exist_ok=True)
            if (is_dlb_data or is_roadtest) and sensor in DLB_TOPICS:
                parse_can_from_pb(file_path, os.path.join(can_path, 'can.json'))
            else:
                _parse_can_json(file_path, os.path.join(can_path, 'can.json'), collection_type, car_type)
        # parse RTK
        elif sensor == 'RTK':
            rtk_path = os.path.join(output_path, "rtk")
            os.makedirs(rtk_path, exist_ok=True)
            _parse_rtk_json(file_path, os.path.join(rtk_path, 'rtk.csv'), collection_type, car_type)
        # parse IMU
        elif sensor == 'IMU':
            imu_path = os.path.join(output_path, "rtk")
            os.makedirs(imu_path, exist_ok=True)
            if (is_dlb_data or is_roadtest) and sensor in DLB_TOPICS:
                parse_imu_from_pb(file_path, os.path.join(imu_path, 'raw_imu.csv'))
            else:
                _parse_imu_json(file_path, os.path.join(imu_path, 'raw_imu.csv'), collection_type, car_type)
        # parse GNSS
        elif sensor == 'GNSS':
            gnss_path = os.path.join(output_path, "rtk")
            os.makedirs(gnss_path, exist_ok=True)
            if (is_dlb_data or is_roadtest) and sensor in DLB_TOPICS:
                parse_gnss_from_pb(file_path, os.path.join(gnss_path, 'raw_rtk.csv'))
            else:
                os.system("cp {0} {1}".format(file_path, os.path.join(gnss_path, 'raw_rtk.csv')))
        # parse FALCON
        elif sensor == 'FALCON':
            if save_lidar_frames:
                falcon_path = os.path.join(output_path, 'FALCON')
                os.makedirs(falcon_path, exist_ok=True)
                parse_falcon_file(file_path, falcon_path)
        elif sensor in HESAI_LIDARS:
            if save_lidar_frames:
                pandar_path = os.path.join(output_path, sensor)
                os.makedirs(pandar_path, exist_ok=True)
                parse_hesai_lidar_point_cloud(clip_path, sensor, pandar_path)
            pass
        # parse video
        elif sensor in CAMERAS:
            meta_file_path = os.path.join(clip_path, 'meta.json')
            camera_path = os.path.join(output_path, "camera_full", sensor)
            os.makedirs(camera_path, exist_ok=True)
            video_timestamp_file_path_prefix = os.path.join(camera_path, file_name.split('/')[-1])
            my_parse_video_time(meta_file_path, file_path, video_timestamp_file_path_prefix)

            if save_video_frames:
                camera_videos.append((sensor, file_path, video_timestamp_file_path_prefix + '.timestamps'))

    if save_video_frames:
        camera_path = os.path.join(output_path, "camera_full")
        os.makedirs(camera_path, exist_ok=True)
        parse_video(camera_videos, camera_path)

    global_localization_file_name = _get_topic_file(meta_info, "/common/localization/global_localization")
    if global_localization_file_name is not None:
        global_localization_file_path = os.path.join(clip_path, global_localization_file_name)
        parse_global_localization_from_pb(global_localization_file_path,
                                          os.path.join(output_path, 'rtk/global_localization.csv'))

    return clip_path, calibration_file_path


def main():
    t0 = time.time()
    parser = parse_args()
    download_and_parse_clip(parser.output_dir, parser.clip_id, parser.sensors,
                            parser.save_lidar_frames, parser.save_video_frames,
                            parser.start_time, parser.end_time)
    print('download completed in {} seconds'.format(time.time() - t0))

if __name__ == "__main__":
    main()
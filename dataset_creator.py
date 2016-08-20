import csv
from multiprocessing import Pool

from config import *
from dataset.file_processing import process_file, create_table_header, write_features, create_pool_input
from dataset.utils import scale_features


if __name__ == '__main__':
    pool = Pool(PROCESSES_NUM)

    if MAX_SPEECH_FILES > 0:
        voiced_writer = csv.writer(open(VOICED_FNAME, 'w'), delimiter=',')
        voiced_writer.writerows([create_table_header(MFCC_NUM)])

    if MAX_NOISE_FILES > 0:
        unvoiced_writer = csv.writer(open(UNVOICED_FNAME, 'w'), delimiter=',')
        unvoiced_writer.writerows([create_table_header(MFCC_NUM)])


    if len(speech_files) > MAX_SPEECH_FILES:
        speech_files = speech_files[:MAX_SPEECH_FILES]

    if len(noise_files) > MAX_NOISE_FILES:
        noise_files = noise_files[:MAX_NOISE_FILES]

    files_counter = 0
    while files_counter < len(speech_files):
        if files_counter + FILES_PER_STEP < len(speech_files):
            features = pool.map(process_file, speech_files[files_counter:files_counter + FILES_PER_STEP])
        else:
            features = pool.map(process_file, speech_files[files_counter:])

        features = scale_features(features)
        write_features(voiced_writer, features, VOICED)
        files_counter += FILES_PER_STEP
    del features

    files_counter = 0
    while files_counter < len(noise_files):
        if files_counter + FILES_PER_STEP < len(noise_files):
            features = pool.map(process_file, noise_files[files_counter:files_counter + FILES_PER_STEP])
        else:
            features = pool.map(process_file, noise_files[files_counter:])

        features = scale_features(features)
        write_features(unvoiced_writer, features, NONE_VOICED)
        files_counter += FILES_PER_STEP
    del features





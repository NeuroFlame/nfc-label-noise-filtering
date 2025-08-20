def convert_fnc_to_features(fnc_path: str, dest_path: str, name: str):
    file_path = os.path.join(fnc_path, f'{name}.mat')
    original_data = load_data_matfile(
        file_path,
        name=[
            SourceDataKeys.SFNC.value,
            SourceDataKeys.FILE_ID.value,
            SourceDataKeys.ANALYSIS_SCORE.value,
        ],
    )
    
    # print(original_data[SourceDataKeys.FILE_ID.value])
    
    label_index=-1
    for index in range(len(original_data[SourceDataKeys.FILE_ID.value])):
        if "diagnosis" in original_data[SourceDataKeys.FILE_ID.value][index].lower():
            label_index = index
            break

    labels = original_data[SourceDataKeys.ANALYSIS_SCORE.value][:,label_index]
    labels = np.reshape(labels, (len(labels), 1))
    fnc_matrices = original_data[SourceDataKeys.SFNC.value]
    
    print("shape of fnc_matrices: ", fnc_matrices.shape)

    features = fnc_matrices.shape[1] * (fnc_matrices.shape[1] - 1) // 2
    subjects_count = fnc_matrices.shape[0]

    source_data = np.zeros((subjects_count, features), dtype=np.float32)

    for i in range(subjects_count):
        fnc_matrix: np.ndarray = fnc_matrices[i]
        # lower triangle matrix without the diagonal
        lower_traiangle = np.tril(fnc_matrix, k=-1)
        lower_traiangle = lower_traiangle[lower_traiangle != 0]
        source_data[i] = lower_traiangle

    source_data = np.hstack((source_data, labels))
    
    output_path = os.path.join(dest_path, f'{name}.mat')
    savemat(output_path, {
        name: source_data
    }, do_compression=True)

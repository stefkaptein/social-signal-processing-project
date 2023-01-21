import pandas as pd
import numpy as np
import random

from source.model.scoring_metrics import get_pk, get_k_kappa, get_windiff

# Arbitrary splits. Tries to keep some instance of all types of files in all of the splits
train_names = """Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017 Bmr001 Bmr002 Bmr005 Bmr007 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr018 Bmr019 Bmr021 Bmr022 Bmr024 Bmr025 Bmr026 Bmr027 Bmr029""".split(" ")
test_names = """Bns001 Bns002 Bns003 Bro003 Bro004 Bro005 Bro007 Bro008 Bro010 Bro011 Bro012 Bro013 Bro014 Bro015 Bro016 Bro017 Bro018 Bro019 Bro021 Bro022 Bro023 Bro024 Bro025 Bro026 Bro027 Bro028 Bsr001 Btr001 Btr002""".split(" ")
# Test and validation are the same for now... because yes
validation_names = """Bns001 Bns002 Bns003 Bro003 Bro004 Bro005 Bro007 Bro008 Bro010 Bro011 Bro012 Bro013 Bro014 Bro015 Bro016 Bro017 Bro018 Bro019 Bro021 Bro022 Bro023 Bro024 Bro025 Bro026 Bro027 Bro028 Bsr001 Btr001 Btr002""".split(" ")

all_names = """Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017 Bmr001 Bmr002 Bmr005 Bmr007 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr018 Bmr019 Bmr021 Bmr022 Bmr024 Bmr025 Bmr026 Bmr027 Bmr029 Bns001 Bns002 Bns003 Bro003 Bro004 Bro005 Bro007 Bro008 Bro010 Bro011 Bro012 Bro013 Bro014 Bro015 Bro016 Bro017 Bro018 Bro019 Bro021 Bro022 Bro023 Bro024 Bro025 Bro026 Bro027 Bro028 Bsr001 Btr001 Btr002""".split(" ")

def test_set_evaluate(model, features, shifts=[], k=None):
    """Input to test trained model for topic segmentation. Returns the results as a dictionary with the pk, windiff,
    and k-kappa results.

    Uses the test set for evaluation, not the valiation set!

    :param model: Model to test. Must have a .predict(X) method for it to work properly. This method
    Should not predict probabilities, but give the actual estimated value
    This method assumes that any fusion has already been done and as such isn't needed
    :param features: Features model was trained on. They're consistent across all datasets used.
    Method assumes that boundary is not included in the list of features, as it's always the target feature
    :param shifts: List containing the nearby sentences that are used. IE [-1, 1] would use the previous
    and the next sentence. Used to transform the input in case it is required. Sentence shifts the model
    used. Important to guarantee consistency
    :param k: Value used for the scoring metrics. In case None, a default is calculated for each of the
    relevant metrics.
    :returns Dictionary containing the three relevant tested metrics"""

    # First step is reading in dataset

    # Given there's no clear boundary split, I'll just be lazy and manually modify the boundary at the
    # end of the dataset to be a 1, as this is the end of a topic. Then concatenate everything.
    # I've seen no explanation on how else to do it, and it seems valid enough.
    base_df = read_in_dataset(shifts, 'test')

    y = base_df['boundary'].to_numpy()
    X = base_df[features]

    # TODO: This should be a nicer way of filling in the nans, depending on the column type.
    #  But for now, this is a good enough starting point
    X.fillna(0, inplace=True)

    hyp_y = model.predict(X)

    results = {'Pk': get_pk(y, hyp_y, k), 'K-k': get_k_kappa(y, hyp_y, k)}
    results['Windiff'] = get_windiff(y, hyp_y, k)

    return results


def test_set_evaluate_multiple(model, features: list, shifts: list =[], k=None):
    """Input to test trained model for topic segmentation. Returns the results as a dictionary with the pk, windiff,
    and k-kappa results for each of the conversations in the test set.

    Uses the test set for evaluation, not the valiation set!

    :param model: Model to test. Must have a .predict(X) method for it to work properly. This method
    Should not predict probabilities, but give the actual estimated value
    This method assumes that any fusion has already been done and as such isn't needed
    :param features: Features model was trained on. They're consistent across all datasets used.
    Method assumes that boundary is not included in the list of features, as it's always the target feature
    :param shifts: List containing the nearby sentences that are used. IE [-1, 1] would use the previous
    and the next sentence. Used to transform the input in case it is required. Sentence shifts the model
    used. Important to guarantee consistency
    :param k: Value used for the scoring metrics. In case None, a default is calculated for each of the
    relevant metrics.
    :returns Dictionary containing the three relevant tested metrics"""
    # First step is reading in dataset

    # Given there's no clear boundary split, I'll just be lazy and manually modify the boundary at the
    # end of the dataset to be a 1, as this is the end of a topic. Then concatenate everything.
    # I've seen no explanation on how else to do it, and it seems valid enough.
    dataset_list = test_names

    pk_values = []
    k_k_values = []
    windiff_values = []

    for elem in dataset_list:
        base_df = pd.read_csv("../results_merged_f0_stds_fixed/" + elem + ".csv", sep=";")
        base_df = transform_rows(base_df, features, shifts)
        y = base_df[['boundary']].to_numpy()
        X = base_df.drop(['boundary'], axis=1)

        # TODO: This should be a nicer way of filling in the nans, depending on the column type.
        #  But for now, this is a good enough starting point
        X.fillna(0, inplace=True)

        hyp_y = model.predict(X)

        pk_values.append(get_pk(y, hyp_y, k))
        k_k_values.append(get_k_kappa(y, hyp_y, k))
        windiff_values.append(get_windiff(y, hyp_y, k))

    results_data = {'Pk': pk_values, 'K-k': k_k_values, 'Windiff': windiff_values}

    return pd.DataFrame(results_data)


def test_set_evaluate_multiple_lstm(model, features, shifts=[0], threshold=0.5, k=None,
                                    location = "../results_merged_f0_stds_fixed/"):
    dataset_list = test_names

    pk_values = []
    k_k_values = []
    windiff_values = []

    # When doing predictions, we only take where shift = 0. This find the location in y, for later use
    target_col = shifts.index(0)

    for elem in dataset_list:
        base_df = pd.read_csv(location + elem + ".csv", sep=";")
        y = base_df['boundary']
        X = base_df[features]

        y = create_3d_df(y, shifts)
        X = create_3d_df(X, shifts)

        # TODO: This should be a nicer way of filling in the nans, depending on the column type.
        #  But for now, this is a good enough starting point
        # X.fillna(0, inplace=True)
        X = np.asarray(X).astype('float32')

        hyp_y = model.predict(X)

        # Now, I need to only take the 0th value, as well as round the parameters...
        hyp_y = hyp_y[:, target_col]
        # Rounding according to a threshold, as well as flattening because of silly numpy stuff
        hyp_y = (hyp_y > threshold).flatten()

        flattened_y = y[:, target_col].flatten()

        pk_values.append(get_pk(flattened_y, hyp_y, k))
        k_k_values.append(get_k_kappa(flattened_y, hyp_y, k))
        windiff_values.append(get_windiff(flattened_y, hyp_y, k))

    results_data = {'Pk': pk_values, 'K-k': k_k_values, 'Windiff': windiff_values}

    return pd.DataFrame(results_data)


def read_in_dataset_lstm(features: list, shifts: list = [-1, 0, 1], to_read = 'train',
                         location = "../results_merged_f0_stds_fixed/"):
    """Selects one of the train, test, or validation dataset, reads them all in, and merges them
    into a larger dataset. This larger dataset is then returned.

    For each of the sub entries, the last boundary is marked as a 1, as it is the end of this set
    of data, compared to the previous one.

    Different than default because instead of adding information to an existing row, it adds a new
    dimension to the dataframe, as well as getting the features in advance

    :param features: the features to use in the dataset. SHOULD NOT INCLUDE BOUNDARY! That gets added
    elsewhere
    :param shifts: Does the relevant sentence shifts asked for. Format is same as transform rows,
    so it's ints explaining the distance from the target sentence. IE -1 means previous one, 2 means
    sentence 2 sentences away. Must also have the target sentence, which is assumed to be 0
    :param to_read: dataset to read in. It's a string, containing either: 'train', 'test', 'validation'.
    If none of these are returned,, the test dataset is returned

    :returns two 3d arrays, one repreneting the X, one representing the y"""

    if to_read == 'train':
        dataset_list = train_names
    elif to_read == 'validation':
        dataset_list = validation_names
    else:
        dataset_list = test_names

    base_df = pd.read_csv(location + dataset_list[0] + ".csv", sep=";")
    base_y = base_df[['boundary']]
    base_x = base_df[features]
    # base_y.iloc[-1] = 1.0

    base_x = create_3d_df(base_x, shifts)
    base_y = create_3d_df(base_y, shifts)

    # First part is to modify the temp_df to be shifted and have the extra dimensions
    # I also have to
    # Then it's just appending it to the base_df
    for i in range(1, len(dataset_list)):
        elem = dataset_list[i]
        temp_df = pd.read_csv(location + elem + ".csv", sep=";")
        temp_y = temp_df[['boundary']]
        temp_x = temp_df[features]
        # temp_y.iloc[-1] = 1.0

        # Just testing that the val gets changed correctly
        # print(temp_y.iloc[-1])

        temp_x = create_3d_df(temp_x, shifts)
        temp_y = create_3d_df(temp_y, shifts)

        base_x = np.concatenate([base_x, temp_x])
        base_y = np.concatenate([base_y, temp_y])

    # Have to transform the X into float, because yes
    return base_x.astype(float), base_y


def create_3d_df(df: pd.DataFrame, shifts: list, fill_nans=True):
    """Receives the original dataframe, and modifies it to have the dimensions as defined by
    shifts. IE it makes it a multidimensional dataframe. Also fills in the generated nan values

    Additionally, it changes nans and modifies the end target. Modifying the end target means
    that if it is the last element of the 0th shift, the boundary gets transformed into a 1

    :param dataframe: Row dataframe that contains all of the information that will be processed
    :param shifts: Represents the surrounding sentences to take. So, -1 means it takes the previous sentence,
    1 means it takes the next sentence, 2 means the sentence 2 steps away, etc. By default, only the previous
    sentence is taken. Also determines the format of the dimensions of the resulting 3d dataset
    :param fill_nans: Determines whether to fill in any nan values generated from the shifts
    """
    df_list = []
    for elem in shifts:
        shifted_df = df.shift(elem).add_suffix(str(elem))
        if fill_nans:
            shifted_df.fillna(0, inplace=True)

        df_list.append(shifted_df)

    temp = np.array(df_list)
    # I have to rotate it to have the shape that Jan asked for
    return np.rot90(temp)


def read_in_dataset(features: list, shifts: list = [-1], to_read = 'train'):
    """Selects one of the train, test, or validation dataset, reads them all in, and merges them
    into a larger dataset. This larger dataset is then returned.

    For each of the sub entries, the last boundary is marked as a 1, as it is the end of this set
    of data, compared to the previous one.

    :param shifts: Does the relevant sentence shifts asked for. Format is same as transform rows,
    so it's ints explaining the distance from the target sentence. IE -1 means previous one, 2 means
    sentence 2 sentences away.
    :param to_read: dataset to read in. It's a string, containing either: 'train', 'test', 'validation'.
    If none of these are returned,, the test dataset is returned

    :returns The relevant combination of the datasets
    """
    if to_read == 'train':
        dataset_list = train_names
    elif to_read == 'validation':
        dataset_list = validation_names
    else:
        dataset_list = test_names

    base_df = pd.read_csv("../results_merged_f0_stds_fixed/" + dataset_list[0] + ".csv", sep=";")
    # Changing the last entry in the base df, to let the system know that it is the
    # end of a topic
    # Last entry is also the boundary, which is why it's -1, -1

    base_df = transform_rows(base_df, features, shifts)

    base_y = base_df[['boundary']]
    base_df = base_df.drop(['boundary'], axis=1)

    for i in range(1, len(dataset_list)):
        elem = dataset_list[i]
        temp_df = pd.read_csv("../results_merged_f0_stds_fixed/" + elem + ".csv", sep=";")

        temp_df = transform_rows(temp_df, features, shifts)

        temp_y = temp_df[['boundary']]
        temp_df = temp_df.drop(['boundary'], axis=1)

        base_df = pd.concat([base_df, temp_df], ignore_index=True)
        base_y = pd.concat([base_y, temp_y], ignore_index=True)

    return base_df, base_y


def read_in_dataset_all_together(features: list, shifts: list = [-1], test_split = 0.3):
    """Selects one of the train, test, or validation dataset, reads them all in, and merges them
    into a larger dataset. This larger dataset is then returned.

    For each of the sub entries, the last boundary is marked as a 1, as it is the end of this set
    of data, compared to the previous one.

    :param features: Features to be used. Should in
    :param shifts: Does the relevant sentence shifts asked for. Format is same as transform rows,
    so it's ints explaining the distance from the target sentence. IE -1 means previous one, 2 means
    sentence 2 sentences away.

    :returns The relevant combination of the datasets
    """
    dataset_list = all_names

    if 'boundary' not in features:
        features.append('boundary')

    train_num_meetings = int(len(dataset_list) * (1-test_split))
    train_selected_meetings = random.sample(dataset_list, train_num_meetings)
    test_selected_meetings = list(set(dataset_list) - set(train_selected_meetings))

    base_df_train = pd.read_csv("../results_merged_f0_stds_fixed/" + train_selected_meetings[0] + ".csv", sep=";")

    base_df_train = transform_rows(base_df_train, features, shifts)

    base_y_train = base_df_train[['boundary']]
    base_df_train = base_df_train.drop(['boundary'], axis=1)

    for i in range(1, len(train_selected_meetings)):
        elem = train_selected_meetings[i]
        temp_df = pd.read_csv("../results_merged_f0_stds_fixed/" + elem + ".csv", sep=";")

        temp_df = transform_rows(temp_df, features, shifts)

        temp_y = temp_df[['boundary']]
        temp_df = temp_df.drop(['boundary'], axis=1)

        base_df_train = pd.concat([base_df_train, temp_df], ignore_index=True)
        base_y_train = pd.concat([base_y_train, temp_y], ignore_index=True)

    base_df_test = pd.read_csv("../results_merged_f0_stds_fixed/" + test_selected_meetings[0] + ".csv", sep=";")

    base_df_test = transform_rows(base_df_test, features, shifts)

    base_y_test = base_df_test[['boundary']]
    base_df_test = base_df_test.drop(['boundary'], axis=1)

    for i in range(1, len(test_selected_meetings)):
        elem = test_selected_meetings[i]
        temp_df = pd.read_csv("../results_merged_f0_stds_fixed/" + elem + ".csv", sep=";")

        temp_df = transform_rows(temp_df, features, shifts)

        temp_y = temp_df[['boundary']]
        temp_df = temp_df.drop(['boundary'], axis=1)

        base_df_test = pd.concat([base_df_test, temp_df], ignore_index=True)
        base_y_test = pd.concat([base_y_test, temp_y], ignore_index=True)

    return base_df_train, base_y_train, base_df_test, base_y_test


def transform_rows(dataframe: pd.DataFrame, features: list, shifts: list = [-1]):
    """Receives the original dataframe, loops through all of the rows and transforms them to
    be valid for a generic classifier.

    Method assumes that the boundary column always exists. When the value is a 1, that means
    it's the end of the topic. Which is the important part for the topic segmentation.

    Method automatically adds boundary to the list of features, because it should always be used

    :param dataframe: Row dataframe that contains all of the information that will be processed
    :param features: The columns that will be kept of the dataframe. Assumed to be list like that contains
    the titles of any features that are to be kept. In case further transformation is required,
    it will be done externally. Boundary should not be included in this list, gets added manually afterwards.
    Only the target sentence boundary is added
    In case features is an empty array, then all of the features get added in
    :param shifts: Represents the surrounding sentences to take. So, -1 means it takes the previous sentence,
    1 means it takes the next sentence, 2 means the sentence 2 steps away, etc. By default, only the previous
    sentence is taken. In case an empty list is passed by, then the dataset is returned as is but with a
    0 added to the end

    :returns Transformed dataframe that already contains the relevant columns. Each feature can
    """

    if len(features) == 0:
        filtered_df = dataframe.drop(['boundary'], axis=1)
    else:
        filtered_df = dataframe[features]

    temp_df = filtered_df.add_suffix('0')
    # Doing a filter on nans, just in case
    temp_df = handle_nas(temp_df)

    # This loop does the shifts, as well as filling in the relevant empty values
    for elem in shifts:
        shifted_df = filtered_df.shift(elem).add_suffix(str(elem))

        # There's an issue specifically with speaker change being casted. I'm just going to manually fix it
        #  up, just because that feels like it's the most logical way of going through it.
        # Shouldn't be done manually, but whatever this works
        dtype_name_to_change = 'speakerChange' + str(elem)
        shifted_df = shifted_df.astype({dtype_name_to_change: 'bool'})
        # Way to keep the nans and keep the rows empty as relevant
        shifted_df = handle_nas(shifted_df)
        temp_df = pd.concat([temp_df, shifted_df], axis=1)

    temp_df['boundary'] = dataframe['boundary']

    return temp_df


def handle_nas(df: pd.DataFrame, default_date='2020-01-01'):
    """
    Helper function to replace nans according to the datatype. Copied from:
    https://stackoverflow.com/questions/59802226/fillna-depending-on-column-type-function

    :param df: a dataframe
    :param default_date: current iterations run_date
    :return: a data frame with replacement of na values as either 0 for numeric fields, 'na' for text and False for bool
    """
    for f in df.columns:

        # integer
        if df[f].dtype == "int":
            df[f] = df[f].fillna(0)

        # dates
        elif df[f].dtype == '<M8[ns]':
            df[f] = df[f].fillna(pd.to_datetime(default_date))

        # boolean
        elif df[f].dtype == 'bool':
            df[f] = df[f].fillna(False)

        # float
        elif df[f].dtype == 'float64':
            df[f] = df[f].fillna(0.0)

        elif df[f].dtype == 'float':
            df[f] = df[f].fillna(0.0)

        # string
        else:
            df[f] = df[f].fillna('na')

    return df


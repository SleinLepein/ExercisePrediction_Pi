from sklearn.preprocessing import MinMaxScaler

def normalize_df(df):
    """ normalizes the dataframe

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    cols_to_norm = [col for col in df.columns if 'Value' in col]
    scaler = MinMaxScaler()
    df.loc[:, cols_to_norm] = scaler.fit_transform(df[cols_to_norm].copy())
    return df

def drop_nothing_rows(df_prediction_samples_normalized, df, list_of_predicted_batches):
    """ Drop all rows that corresponde to a window that doesn't contain an exercise (all nothing windows)

    Parameters
    ----------
    df_prediction_samples_normalized : pd.DataFrame
        df with normalized data
    df : pd.DataFrame
        df that is not normalized
    list_of_predicted_batches : list
        list that contains all the predicted batches and there tags (exercise or nothing)

    Returns
    -------
    pd.DataFrame
    """
    nothing_indices = []
    not_nothing_tagged = []
    for i,x in enumerate(list_of_predicted_batches):
        if x[0] == 'nothing':
            nothing_indices.append(i)
        else:
            not_nothing_tagged.append(x[0])
    # .index does not work here ???
    return df_prediction_samples_normalized.drop(df.index[nothing_indices], inplace=True), not_nothing_tagged
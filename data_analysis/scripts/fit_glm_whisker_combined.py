import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import shutil
from sklearn.linear_model import ElasticNet
from time import time


def get_x_y(mouse, plane, session, base_dir, glm_type='whisker_combined',
            standardize_traces=True, standardize_features=True):
    
    roi_dir = base_dir / f'{mouse:03}/plane_{plane}/{session:03}/plane0/roi'
    glm_dir = roi_dir / f'glm/{glm_type}'
    design_df = pd.read_pickle(glm_dir / f'design_{glm_type}.pkl')

    spks = np.load(roi_dir / 'spks_reduced.npy')
    iscell = np.load(roi_dir / 'iscell.npy')
    cell_inds = np.where(iscell[:,0]==1)[0]
    spks = spks[cell_inds,:]
    norm_spks = (spks - spks.mean(axis=1)[:,np.newaxis]) / spks.std(axis=1)[:,np.newaxis]
    ophys_frametime = pd.read_pickle(roi_dir / 'refined_frame_time.pkl')
    assert len(ophys_frametime) == spks.shape[1]

    # filter out rows from design_df
    # those with NaN values
    # those that are not in ophys_frametime (trialNum, frame_index)
    keep_ind = np.where(np.isnan(np.sum(design_df.values, axis=1).astype(float))==False)[0]
    filtered_design_df = design_df.iloc[keep_ind]
    if standardize_features:
        feature_namebase_to_standardize = ['touch_count', 'num_lick', 'num_whisks', 'midpoint', 'amplitude']
            # Other 11 whisker features are already standardized
        feature_names_to_standardize = [design_column for design_column in filtered_design_df.columns if any([namebase in design_column for namebase in feature_namebase_to_standardize])]
        for feature_name in feature_names_to_standardize:
            design_df[feature_name] = (design_df[feature_name] - design_df[feature_name].mean()) / design_df[feature_name].std()
    filtered_design_df = filtered_design_df.query('trialNum in @ophys_frametime.trialNum and frame_index in @ophys_frametime.frame_index')
    filtered_design_df = filtered_design_df.reset_index(drop=True)
    assert len(filtered_design_df.frame_index.unique()) == len(filtered_design_df)
    assert np.isin(filtered_design_df.frame_index.values, ophys_frametime.frame_index.values).all()

    spks_frame_inds = np.where(np.isin(ophys_frametime.frame_index.values, filtered_design_df.frame_index.values))[0]
    assert len(spks_frame_inds) == len(filtered_design_df)
    if standardize_traces:
        spks = norm_spks
    traces = spks[:,spks_frame_inds].T 
    # Now traces are in shape of (n_frames, n_cells)

    # Standardization
    # No change in touch, lick, reward, sound, num_whisks
    # Just for amplitude and midpoint
    standardized_names = [key for key in filtered_design_df.keys() if ('midpoint' in key) or ('amplitude' in key)]
    for key in standardized_names:
        filtered_design_df[key] = (filtered_design_df[key] - filtered_design_df[key].mean()) / filtered_design_df[key].std()

    # feature names
    whisker_feature_names_base = ['theta_onset', 'phi_onset', 'kappaH_onset', 'kappaV_onset',
        'arc_length_onset', 'touch_count', 'delta_theta', 'delta_phi',
        'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']
    whisker_feature_names = [key for key in filtered_design_df.keys() if sum([wfnb in key for wfnb in whisker_feature_names_base])==1]
    whisking_names = [key for key in filtered_design_df.keys() if ('num_whisks' in key) or ('midpoint' in key) or ('amplitude' in key)]
    lick_names = [key for key in filtered_design_df.keys() if 'num_lick' in key]
    sound_names = [key for key in filtered_design_df.keys() if 'pole_in_frame' in key or 'pole_out_frame' in key]
    reward_names = [key for key in filtered_design_df.keys() if 'first_reward_lick' in key]

    # Adding the bias column
    X = np.hstack((np.ones((len(filtered_design_df),1)), filtered_design_df[whisker_feature_names + whisking_names + lick_names + sound_names + reward_names].values)).astype(float)

    # Turning into xarray
    x = np.hstack((np.ones((len(filtered_design_df),1)), filtered_design_df[whisker_feature_names + whisking_names + lick_names + sound_names + reward_names].values)).astype(float)
    X = xr.DataArray(x, dims=('index', 'feature'), 
                        coords={'index':filtered_design_df.index.values,
                                'feature':['intercept'] + whisker_feature_names + whisking_names + lick_names + sound_names + reward_names})
    traces = xr.DataArray(traces, dims=('index', 'cell_id'),
                        coords={'index':filtered_design_df.index.values,
                                'cell_id':cell_inds})
    return X, traces, filtered_design_df


def stratify(input_inds, n_folds=5):
    ''' get all the combination of stratification
    '''
    inds = input_inds.copy()
    np.random.shuffle(inds)
    groups = np.array_split(inds, n_folds)

    assert len(groups) == n_folds
    assert len(np.concatenate(groups)) == len(inds)

    return groups


def get_stratified_frame_indice(mouse, plane, session, base_dir, filtered_design_df, n_folds=5):
    # stratification
    # 80% for training, 20% for testing
    # based on trials - touch, angles, task (correct, wrong, miss)
    # within each stratification, randomly divide into 5 folds (of all frames)
    # Do this once for finding lambda, and one for fitting the model
    filtered_design_df.reset_index(drop=True, inplace=True)

    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    behavior_frametime = pd.read_pickle(plane_dir / f'JK{mouse:03}_S{session:02}_plane{plane}_frame_whisker_behavior.pkl')
    trialNums = filtered_design_df.trialNum.unique()
    behavior_frametime = behavior_frametime.query('trialNum in @trialNums')

    touch_trialNums = behavior_frametime.query("touch_count > 0").trialNum.unique()
    nontouch_trialNums = np.setdiff1d(behavior_frametime.trialNum.unique(), touch_trialNums)
    trialNum_strat_touch = [touch_trialNums, nontouch_trialNums]

    trialNum_strat_angle = []
    angles = behavior_frametime['pole_angle'].unique()
    for angle in angles:
        trialNum_strat_angle.append(behavior_frametime.query('pole_angle == @angle')['trialNum'].unique())

    result_variables = ['correct', 'wrong', 'miss']
    trialNum_strat_result = []
    for result in result_variables:
        trialNum_strat_result.append(behavior_frametime[behavior_frametime[result]].trialNum.unique())

    lambda_groups = []
    fit_groups = []
    all_tns = 0
    for touch_tns in trialNum_strat_touch:
        for angle_tns in trialNum_strat_angle:
            for result_tns in trialNum_strat_result:
                start_tns = np.intersect1d(np.intersect1d(touch_tns, angle_tns), result_tns)
                all_tns += len(start_tns)

                if len(start_tns) > 0:
                    trace_inds = filtered_design_df.query('trialNum in @start_tns').index.values
                    temp_lambda_groups = stratify(trace_inds, n_folds=n_folds)
                    if len(lambda_groups) == 0:
                        lambda_groups = temp_lambda_groups
                    else:
                        lambda_groups = [np.append(lambda_groups[i], temp_lambda_groups[i]) for i in range(n_folds)]
                    temp_fit_groups = stratify(trace_inds, n_folds=n_folds)
                    if len(fit_groups) == 0:
                        fit_groups = temp_fit_groups
                    else:
                        fit_groups = [np.append(fit_groups[i], temp_fit_groups[i]) for i in range(n_folds)]
                    
    assert all_tns == len(filtered_design_df.trialNum.unique())
    # print(len(np.unique(np.concatenate(lambda_groups))), len(np.unique(np.concatenate(fit_groups))), len(filtered_design_df))
    assert len(np.unique(np.concatenate(lambda_groups))) == len(np.unique(np.concatenate(fit_groups))) == len(filtered_design_df)
    assert len(fit_groups) == len(lambda_groups) == n_folds

    return lambda_groups, fit_groups


def fit_glm_gaussian_L2(traces, X, lam):
    '''Fitting Gaussian GLM with L2 regularization
    traces: 2darray (n_frames, n_cells)
    X: n_frames x n_features
    lam: regularization parameter
    ''' 
    w = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * np.eye(X.shape[-1])),
               np.dot(X.T, traces.values))
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    if len(traces.cell_id.values.shape)==0:
        cell_id = np.array([int(traces.cell_id.values)])
    else:
        cell_id = traces.cell_id.values
    W = xr.DataArray(w, dims=['feature', 'cell_id'],
                     coords={'feature': X.feature.values,
                             'cell_id': cell_id})
    return W


def find_lambda(traces, X, lambda_groups, lam_grid=np.geomspace(0.1, 10000, 100)):
    '''Finding lambda for L2 regularization
    traces: xarray (n_frames, n_cells)
    X: xarray (n_frames x n_features)
    lam_grid: grid of lambda values
    '''
    n_lams = len(lam_grid)
    n_cells = traces.shape[1]
    varexps = np.zeros((n_cells, n_lams, len(lambda_groups)))
    for i, lam in enumerate(lam_grid):
        for j in range(len(lambda_groups)):
            train_inds = np.concatenate([lambda_groups[k] for k in np.setdiff1d(range(len(lambda_groups)), j)])
            test_inds = lambda_groups[j]
            W = fit_glm_gaussian_L2(traces.isel(index=train_inds), X.isel(index=train_inds), lam)
            _, ve = variance_ratio(traces.isel(index=test_inds), W, X.isel(index=test_inds))
            varexps[:, i, j] = ve
    varexps[np.isinf(varexps)] = np.nan
    mean_varexps = np.nanmean(varexps, axis=2)
    return lam_grid[np.argmax(mean_varexps, axis=1)], mean_varexps


def variance_ratio(traces, W, X): 
    '''
    Computes the fraction of variance in traces explained by the linear model Y = X*W
    
    traces: xarray (n_frames, n_cells)
    W: xarray (n_features, n_cells)
    X: xarray (n_frames, n_features)
    '''
    if len(traces.shape) == 1:
        assert len(traces) == X.shape[0]
        trace = traces.values[:,np.newaxis]
    elif len(traces.shape) == 2:
        trace = traces.values
    pred = X.values @ W.values
    ve = 1 - (np.var(trace - pred, axis=0) / np.var(trace, axis=0))
    return pred, ve


def fit_model(X, traces, fit_groups, lambdas,
              model='ridge'):
    ''' Fitting the model with cell-specific lambda
    X: xarray (n_frames x n_features)
    traces: xarray (n_frames, n_cells)
    fit_groups: list of indices for cross-validation
    lambdas: 1d array of lambda values

        Returns:
        predicted_final: xarray (n_frames, n_cells)
        ve_model_final: xarray (n_cells)
        W_final: xarray (n_features, n_cells)
        predicted_fold: xarray (n_frames, n_cells)
        varexp_fold: xarray (n_cells, n_folds)
        W_all: xarray (n_features, n_cells, n_folds)
    '''
    assert model in ['ridge', 'lasso']
    predicted_fold = np.zeros(traces.shape)
    varexp_fold = np.zeros((traces.shape[1], len(fit_groups)))
    W_all = xr.DataArray(np.zeros((X.shape[-1], traces.shape[1], len(fit_groups))),
                            dims=('feature', 'cell_id', 'fold'),
                            coords={'feature': X.feature.values,
                                    'cell_id': traces.cell_id.values,
                                    'fold': [f'fold_{i}' for i in np.arange(len(fit_groups))]})
    for fi in range(len(fit_groups)):
        train_inds = np.concatenate([fit_groups[k] for k in np.setdiff1d(range(len(fit_groups)), fi)])
        test_inds = fit_groups[fi]
        for ci in range(traces.shape[1]):
            lam = lambdas[ci]
            if model == 'ridge':
                W_temp = fit_glm_gaussian_L2(traces.isel(index=train_inds,cell_id=ci), X.isel(index=train_inds), lam)
                pred, ve = variance_ratio(traces.isel(index=test_inds,cell_id=ci), W_temp, X.isel(index=test_inds))
            else:
                glm = fit_glm_lasso_single(traces.isel(index=train_inds,cell_id=ci), X.isel(index=train_inds), lam)
                W_temp = glm.coef_
                pred, ve = glm_variance_ratio(glm, traces.isel(index=test_inds,cell_id=ci), X.isel(index=test_inds))
            W_all[:,ci,fi] = W_temp.squeeze()
            predicted_fold[test_inds,ci] = pred.reshape((len(test_inds),))
            varexp_fold[ci, fi] = ve
    varexp_fold = xr.DataArray(varexp_fold,
                                dims=('cell_id', 'fold'),
                                coords={'cell_id':traces.cell_id.values,
                                        'fold': [f'fold_{i}' for i in np.arange(len(fit_groups))]})
    predicted_fold = xr.DataArray(predicted_fold, dims=('index', 'cell_id'),
                             coords={'index':traces.index.values,
                                     'cell_id':traces.cell_id.values})
    
    # adding final model of a cell  
    W_final = W_all.mean(dim='fold')
    pred_final, ve_final = variance_ratio(traces, W_final, X)
    predicted_final = xr.DataArray(pred_final, dims=('index', 'cell_id'),
                             coords={'index':traces.index.values,
                                     'cell_id':traces.cell_id.values})
    ve_final = xr.DataArray(ve_final, dims=('cell_id'),
                             coords={'cell_id':traces.cell_id.values})

    return predicted_final, ve_final, W_final, predicted_fold, varexp_fold, W_all


def drop_feature_and_fit(X, traces, lambda_groups, fit_groups, feature_names,
                         model='ridge'):
    '''Drop each features and fit the model
    Re-run finding lambda as well
    '''
    assert model in ['ridge', 'lasso']
    n_cells = traces.shape[1]
    n_features = len(feature_names)
    varexp_drop = xr.DataArray(np.zeros((n_cells, n_features)),
							dims=('cell_id', 'dropped_feature'),
							coords={'cell_id':traces.cell_id.values,
									'dropped_feature': list(feature_names.keys())})
    predicted_drop = xr.DataArray(np.zeros((*traces.shape, n_features)),
									dims=('index', 'cell_id', 'dropped_feature'),
									coords={'index':traces.index.values,
											'cell_id':traces.cell_id.values,
											'dropped_feature': list(feature_names.keys())})
    W_drop = xr.DataArray(np.zeros((X.shape[-1], n_cells, n_features)),
							dims=('feature', 'cell_id', 'dropped_feature'),
							coords={'feature': X.feature.values,
									'cell_id':traces.cell_id.values,
									'dropped_feature': list(feature_names.keys())})
    lambdas_drop = xr.DataArray(np.zeros((n_cells, n_features)),
							dims=('cell_id', 'dropped_feature'),
							coords={'cell_id':traces.cell_id.values,
									'dropped_feature': list(feature_names.keys())})
    for fni, key in enumerate(feature_names.keys()):
        X_temp = X.sel(feature=np.setdiff1d(X.feature.values, feature_names[key]))
        if model == 'ridge':
            lam_d, _ = find_lambda(traces, X_temp, lambda_groups)
            
        else:
            lam_d, _ = lasso_find_lambda(traces, X_temp, lambda_groups)
            # W_d = fit_glm_lasso(traces, X_temp, lam_d)
            # pred_d, ve_d = variance_ratio(traces, W_d, X_temp)
        pred_d, ve_d, W_d, _, _, _ = fit_model(X_temp, traces, fit_groups, lam_d, model=model)
        varexp_drop[:,fni] = ve_d
        predicted_drop[:,:,fni] = pred_d
        matching_indices = {'feature': W_d['feature'], 'cell_id': W_d['cell_id'], 'dropped_feature': key}
        W_drop.loc[matching_indices] = W_d
        lambdas_drop[:,fni] = lam_d

    varexp_drop = xr.DataArray(varexp_drop,
                            dims=('cell_id', 'dropped_feature'),
                            coords={'cell_id':traces.cell_id.values,
                                    'dropped_feature': list(feature_names.keys())})
    predicted_drop = xr.DataArray(predicted_drop,
                                    dims=('index', 'cell_id', 'dropped_feature'),
                                    coords={'index':traces.index.values,
                                            'cell_id':traces.cell_id.values,
                                            'dropped_feature': list(feature_names.keys())})
    W_drop = xr.DataArray(W_drop,
                            dims=('feature', 'cell_id', 'dropped_feature'),
                            coords={'feature': X.feature.values,
                                    'cell_id':traces.cell_id.values,
                                    'dropped_feature': list(feature_names.keys())})
    lambdas_drop = xr.DataArray(lambdas_drop,
                            dims=('cell_id', 'dropped_feature'),
                            coords={'cell_id':traces.cell_id.values,
                                    'dropped_feature': list(feature_names.keys())})
    return varexp_drop, predicted_drop, W_drop, lambdas_drop


def run_glm_and_save(mouse, plane, session, base_dir, glm_type='whisker_combined'):
    print(f'Fitting GLM for {mouse:03} plane {plane} session {session:03} - {glm_type}')
    roi_dir = base_dir / f'{mouse:03}/plane_{plane}/{session:03}/plane0/roi'
    glm_dir = roi_dir / f'glm/{glm_type}'
    # requires preprocessed design matrix in the glm directory
    X, traces, filtered_design_df = get_x_y(mouse, plane, session, base_dir, glm_type=glm_type)

    # stratification and splitting
    lambda_groups, fit_groups = get_stratified_frame_indice(mouse, plane, session, base_dir, filtered_design_df)

    # fitting the full model
    lambdas, _ = find_lambda(traces, X, lambda_groups)
    predicted_final, ve_model_final, W_final, predicted_fold, varexp_model_fold, W_fold = fit_model(X, traces, fit_groups, lambdas,
                                                                                                    model='ridge')

    # drop each feature and fit the partial models
    whisker_feature_names_base = ['theta_onset', 'phi_onset', 'kappaH_onset', 'kappaV_onset',
        'arc_length_onset', 'touch_count', 'delta_theta', 'delta_phi',
        'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']
    whisker_feature_names = [key for key in filtered_design_df.keys() if sum([wfnb in key for wfnb in whisker_feature_names_base])==1]
    whisking_names = [key for key in filtered_design_df.keys() if ('num_whisks' in key) or ('midpoint' in key) or ('amplitude' in key)]
    lick_names = [key for key in filtered_design_df.keys() if 'num_lick' in key]
    sound_names = [key for key in filtered_design_df.keys() if 'pole_in_frame' in key or 'pole_out_frame' in key]
    reward_names = [key for key in filtered_design_df.keys() if 'first_reward_lick' in key]
    feature_names = {'whisker_feature': whisker_feature_names,
                    'whisking': whisking_names,
                    'lick': lick_names,
                    'sound': sound_names,
                    'reward': reward_names}
    varexp_drop, predicted_drop, W_drop, lambdas_drop = drop_feature_and_fit(X, traces, lambda_groups, fit_groups, feature_names)

    # save the results
    dataset = xr.Dataset({'traces': traces,
                            'X': X,
                            'varexp_model_final': ve_model_final,
                            'predicted_final': predicted_final,
                            'W_final': W_final,
                            'predicted_fold': predicted_fold,
                            'varexp_model_fold': varexp_model_fold,
                            'W_fold': W_fold,
                            'lambdas': lambdas,
                            'varexp_drop': varexp_drop,
                            'predicted_drop': predicted_drop,
                            'W_drop': W_drop,
                            'lambdas_drop': lambdas_drop,})
    dataset.to_netcdf(glm_dir / 'glm_result.nc')
    # save the splits for replication
    split = {'lambda_groups': lambda_groups,
            'fit_groups': fit_groups}
    np.save(glm_dir / 'glm_split.npy', split)
    


############
## Lasso
############
def fit_glm_lasso_single(trace, X, lam, fit_intercept=False):
    if fit_intercept == False:
        assert 'intercept' in X.feature.values
    glm = ElasticNet(alpha=lam, l1_ratio=0.95, fit_intercept=fit_intercept)
    glm.fit(X, trace)
    return glm


def glm_variance_ratio(glm, trace, X):
    pred = glm.predict(X)
    ve = 1 - np.var(trace - pred, axis=0) / np.var(trace, axis=0)
    return pred, ve


def fit_glm_lasso(traces, X, lamdas, fit_intercept=False):
    for ci in range(traces.shape[1]):
        glm = fit_glm_lasso_single(traces.isel(cell_id=ci), X, lamdas[ci], fit_intercept=fit_intercept)
        if ci == 0:
            W = xr.DataArray(glm.coef_[:,np.newaxis],
                            dims=('feature', 'cell_id'),
                            coords={'feature': X.feature.values,
                                    'cell_id': [traces.cell_id.values[ci]]})
        else:
            W = xr.concat([W, xr.DataArray(glm.coef_[:,np.newaxis],
                            dims=('feature', 'cell_id'),
                            coords={'feature': X.feature.values,
                                    'cell_id': [traces.cell_id.values[ci]]})], dim='cell_id')
    return W


def lasso_find_lambda(traces, X, lambda_groups, lam_grid=np.logspace(-4, 0, 20)):
    if 'intercept' not in X.feature.values:
        X = xr.concat([X, xr.DataArray(np.ones((X.shape[0], 1)), dims=('index', 'feature'), coords={'index': X.index.values, 'feature': ['intercept']})], dim='feature')
    num_lam_splits = len(lambda_groups)
    n_cells = traces.shape[1]
    n_lams = len(lam_grid)
    varexps = np.zeros((n_cells, n_lams, num_lam_splits))
    for lam_i, lam in enumerate(lam_grid):
        for spi in range(num_lam_splits):
            train_inds = np.concatenate([lambda_groups[i] for i in range(num_lam_splits) if i != spi])
            test_inds = lambda_groups[spi]
            X_train = X.isel(index=train_inds)
            X_test = X.isel(index=test_inds)
            y_train = traces.isel(index=train_inds)
            y_test = traces.isel(index=test_inds)
            glm = ElasticNet(alpha=lam, l1_ratio=0.95, fit_intercept=False)
            glm.fit(X_train, y_train)
            y_pred = glm.predict(X_test)
            varexps[:, lam_i, spi] = 1 - np.var(y_test - y_pred, axis=0) / np.var(y_test, axis=0)
    varexps[np.isinf(varexps)] = np.nan
    mean_varexps = np.nanmean(varexps, axis=2)
    lambdas = lam_grid[np.argmax(mean_varexps, axis=1)]
    return lambdas, mean_varexps


def run_lasso_glm_and_save(mouse, plane, session, base_dir, ridge_dir=None, glm_type='whisker_combined'):
    glm_dir = base_dir / 'results/neuron_glm/lasso/whisker_combined'
    glm_save_fn = glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_result.nc'
    split_save_fn = glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_split.npy'
    if (glm_save_fn.exists()) and (ridge_dir is not None):
        print(f'Lasso GLM already processed: {mouse:03} plane {plane} session {session:03} - {glm_type}')
        if not split_save_fn.exists():
            split_fn = ridge_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_split.npy'
            # copy split_fn to split_save_fn
            shutil.copy(split_fn, split_save_fn)
    else:
        print(f'Fitting lasso GLM for {mouse:03} plane {plane} session {session:03} - {glm_type}')
        if ridge_dir is not None:
            # Load design matrix and splits from the ridge_dir
            split_fn = ridge_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_split.npy'
            split = np.load(split_fn, allow_pickle=True).item()
            rigde_result_fn = ridge_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_result.nc'
            with xr.open_dataset(rigde_result_fn) as ds:
                ridge_result = ds.load()
            traces = ridge_result['traces']
            X = ridge_result['X']
            lambda_groups = split['lambda_groups']
            fit_groups = split['fit_groups']
        else:
            # requires preprocessed design matrix in the glm directory
            X, traces, filtered_design_df = get_x_y(mouse, plane, session, base_dir, glm_type=glm_type)

            # stratification and splitting
            lambda_groups, fit_groups = get_stratified_frame_indice(mouse, plane, session, base_dir, filtered_design_df)
        

        # fitting the full model
        lambdas, _ = lasso_find_lambda(traces, X, lambda_groups)
        predicted_final, ve_model_final, W_final, predicted_fold, varexp_model_fold, W_fold = fit_model(X, traces, fit_groups, lambdas,
                                                                                                        model='lasso')

        # drop each feature and fit the partial models
        whisker_feature_names_base = ['theta_onset', 'phi_onset', 'kappaH_onset', 'kappaV_onset',
            'arc_length_onset', 'touch_count', 'delta_theta', 'delta_phi',
            'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']
        whisker_feature_names = [key for key in X.feature.values if sum([wfnb in key for wfnb in whisker_feature_names_base])==1]
        whisking_names = [key for key in X.feature.values if ('num_whisks' in key) or ('midpoint' in key) or ('amplitude' in key)]
        lick_names = [key for key in X.feature.values if 'num_lick' in key]
        sound_names = [key for key in X.feature.values if 'pole_in_frame' in key or 'pole_out_frame' in key]
        reward_names = [key for key in X.feature.values if 'first_reward_lick' in key]
        feature_names = {'whisker_feature': whisker_feature_names,
                        'whisking': whisking_names,
                        'lick': lick_names,
                        'sound': sound_names,
                        'reward': reward_names}
        varexp_drop, predicted_drop, W_drop, lambdas_drop = drop_feature_and_fit(X, traces, lambda_groups, fit_groups, feature_names,
                                                                                model='lasso')

        # save the results
        dataset = xr.Dataset({'traces': traces,
                                'X': X,
                                'varexp_model_final': ve_model_final,
                                'predicted_final': predicted_final,
                                'W_final': W_final,
                                'predicted_fold': predicted_fold,
                                'varexp_model_fold': varexp_model_fold,
                                'W_fold': W_fold,
                                'lambdas': lambdas,
                                'varexp_drop': varexp_drop,
                                'predicted_drop': predicted_drop,
                                'W_drop': W_drop,
                                'lambdas_drop': lambdas_drop,})
        dataset.to_netcdf(glm_save_fn)
        # save the splits for replication
        split = {'lambda_groups': lambda_groups,
                'fit_groups': fit_groups}
        np.save(split_save_fn, split)    


############
### Post hoc analysis
############
    
## Unique contribution 
def get_unique_contribution(mouse, plane, session, glm_dir, model):
    assert model in ['ridge', 'lasso']
    with xr.open_dataset(glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_result.nc') as glm_result:
        glm_result.load()
    glm_split = np.load(glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_split.npy', allow_pickle=True).item()

    whisker_feature_names_base = ['theta_onset', 'phi_onset', 'kappaH_onset', 'kappaV_onset',
        'arc_length_onset', 'touch_count', 'delta_theta', 'delta_phi',
        'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']
    whisker_feature_dict = {key: [value for value in glm_result.X.feature.values if key in value] for key in whisker_feature_names_base}

    varexp_drop, predicted_drop, W_drop, lambdas_drop = \
        drop_feature_and_fit(glm_result.X, glm_result.traces, 
                            glm_split['lambda_groups'], glm_split['fit_groups'],
                            whisker_feature_dict, model=model)
    unique_contribution = (glm_result.varexp_model_final - 
                           varexp_drop) / glm_result.varexp_model_final
    return unique_contribution

## Feature importance
def get_whisker_feature_importance(mouse, plane, session, glm_dir):
    with xr.open_dataset(glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_result.nc') as glm_result:
        glm_result.load()

    whisker_feature_names_base = ['theta_onset', 'phi_onset', 'kappaH_onset', 'kappaV_onset',
        'arc_length_onset', 'touch_count', 'delta_theta', 'delta_phi',
        'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']
    whisker_feature_dict = {key: [value for value in glm_result.X.feature.values if key in value] for key in whisker_feature_names_base}

    ve_drop = []
    for key, value in whisker_feature_dict.items():
        X_temp = glm_result.X.copy()
        X_temp.loc[{'feature': value}] = 0
        _, ve = variance_ratio(glm_result.traces, glm_result.W_final, X_temp)
        ve_drop.append(ve)
    ve_drop = xr.DataArray(np.array(ve_drop), dims=('feature', 'cell_id'), 
                       coords={'cell_id': glm_result.traces.cell_id.values,
                               'feature': list(whisker_feature_dict.keys())})    
    return ve_drop


def run_glm_posthoc(mouse, plane, session, base_dir, glm_type='whisker_combined', model='ridge'):
    assert model in ['ridge', 'lasso']
    print(f'{glm_type} GLM posthoc for {mouse:03} plane {plane} session {session:03} - {model}, {glm_type}')
    # roi_dir = base_dir / f'{mouse:03}/plane_{plane}/{session:03}/plane0/roi'
    # glm_dir = roi_dir / f'glm/{glm_type}'
    glm_dir = base_dir / f'results/neuron_glm/{model}/{glm_type}'
    save_fn = glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_posthoc.nc'
    if save_fn.exists():
        print(f'{model} GLM posthoc already processed: {mouse:03} plane {plane} session {session:03} - {model}, {glm_type}')
    else:
        t0 = time()
        if model == 'ridge':
            unique_contribution = get_unique_contribution(mouse, plane, session, glm_dir, model=model)
            feature_importance = get_whisker_feature_importance(mouse, plane, session, glm_dir)

            dataset = xr.Dataset({'unique_contribution': unique_contribution,
                                    'feature_importance': feature_importance})
        else:
            feature_importance = get_whisker_feature_importance(mouse, plane, session, glm_dir)
            dataset = xr.Dataset({'feature_importance': feature_importance})
        dataset.to_netcdf(glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_posthoc.nc')
        t1 = time()
        print(f'{(t1-t0)/60:.1f} min elapsed for {mouse:03} plane {plane} session {session:03}')
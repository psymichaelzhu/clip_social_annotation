# %%  regression analysis
def prepare_regression_data(roi, candidate_list, candidate_rsm_dict, neural_rsm):
    """Prepare data for regression analysis by extracting and standardizing RSM data
    Args:
        roi (str): ROI name in 'roi_side' format
        candidate_list (dict): Dictionary specifying which RSMs to use from each module
        candidate_rsm_dict (dict): Dictionary of dictionaries containing candidate RSMs
        neural_rsm (dict): Dictionary containing neural RSMs
    Returns:
        df (pd.DataFrame): DataFrame with RSM values formatted for regression
    """
    def standardize(x,method="rank"):
        """Standardize array by subtracting mean and dividing by std
        """
        if method=="rank":
            return pd.Series(x).rank().values
        elif method=="zscore":
            return (x - np.mean(x)) / np.std(x)
        else:
            return x
    
    # Get lower triangular indices
    n = len(neural_rsm['sub01'][roi])
    tril_idx = np.tril_indices(n, k=-1)
    
    # Extract neural data for all subjects
    neural_data = {}
    for sub in neural_rsm.keys():
        if sub != 'group':
            neural_vec = neural_rsm[sub][roi][tril_idx]
            # Standardize within each subject
            neural_data[sub] = standardize(neural_vec)
    
    # Extract candidate features
    feature_data = {}
    for module, rsm_list in candidate_list.items():
        for rsm_name in rsm_list:
            feature_vec = candidate_rsm_dict[module][rsm_name][tril_idx]
            # Standardize each feature
            feature_data[f"{rsm_name}"] = standardize(feature_vec)
            
    # Create pairs of video indices
    video1_idx = []
    video2_idx = []
    for i, j in zip(*tril_idx):
        video1_idx.append(i)
        video2_idx.append(j)
        
    # Build dataframe
    df_list = []
    for sub in neural_data.keys():
        sub_df = pd.DataFrame({
            'sub': sub,
            'video1': video1_idx,
            'video2': video2_idx,
            'roi': roi,
            'neural': neural_data[sub]
        })
        for feat_name, feat_data in feature_data.items():
            sub_df[feat_name] = feat_data
        df_list.append(sub_df)
    df = pd.concat(df_list, ignore_index=True)
    print(df.head())
    print(df.shape)#n_sub (4) x n_video_pair (244 x 243/2)
    return df

def fit_regression_model(df, multiple_comparison='bonferroni', if_display=True):
    """Fit regression model to predict neural RSM values from candidate features
    Args:
        df (pd.DataFrame): DataFrame with neural and feature data
        if_display (bool): Whether to display regression coefficients plot
    Returns:
        dict: Dictionary containing regression coefficients
    """
    from sklearn.linear_model import LinearRegression
    
    # Prepare X and y
    feature_cols = [col for col in df.columns if col not in ['sub', 'video1', 'video2', 'roi', 'neural']]
    X = df[feature_cols]
    y = df['neural']
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    coef_dict = {'intercept': model.intercept_}
    for feat, coef in zip(feature_cols, model.coef_):
        coef_dict[feat] = coef
        
    if if_display:
        # Sort coefficients to match mixed model order
        feature_cols = [col for col in df.columns if col not in ['sub', 'video1', 'video2', 'roi', 'neural']]
        coef_items = [(feat, coef_dict[feat]) for feat in feature_cols]
        
        plt.figure(figsize=(12,6))
        plt.bar([x[0] for x in coef_items], [x[1] for x in coef_items])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Regression Coefficients for {df.roi.iloc[0]}', fontsize=22)
        plt.xlabel('CandidateFeatures', fontsize=18)
        plt.ylabel('Beta', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.show()
    return coef_dict

def fit_mixed_model(df, multiple_comparison='bonferroni',n_total_comparison=None,if_display=True):
    """Fit mixed effects model to predict neural RSM values from candidate features with random subject effects
    Args:
        df (pd.DataFrame): DataFrame with neural and feature data
        if_display (bool): Whether to display regression results
    Returns:
        dict: Dictionary containing fixed effects, random effects, and model summary
    """
    import statsmodels.formula.api as smf
    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    # Fit mixed effects model with random intercept for subjects
    feature_cols = [col for col in df.columns if col not in ['sub', 'video1', 'video2', 'roi', 'neural']]
    feature_formula = ' + '.join(feature_cols)
    formula = f"neural ~ {feature_formula}"
    
    mixed_model = smf.mixedlm(formula, df, groups=df["sub"], re_formula="1")
    mixed_result = mixed_model.fit()
    # multiple comparison correction
    #bonferroni correction, n_total_comparison is the total number of comparisons; I did it manually here because the direct use of multipletests() only works for one iteration
    qvalues = mixed_result.pvalues*n_total_comparison
    
    # Get fixed effects coefficients with confidence intervals
    fixed_effects = pd.DataFrame({
        'coef': mixed_result.fe_params,
        'std': mixed_result.bse_fe,
        'pvalue': mixed_result.pvalues,
        'qvalue': qvalues,
        'ci_lower': mixed_result.conf_int()[0],
        'ci_upper': mixed_result.conf_int()[1],
        'tvalue': mixed_result.tvalues
    })
    
    # Store results
    results_dict = {
        'fixed_effects': fixed_effects,
        'random_effects': mixed_result.random_effects,
        'model_summary': mixed_result.summary()
    }
    
    if if_display:
        print("\nMixed Effects Model Results:")
        print(mixed_result.summary())
        # Plot fixed effects coefficients with confidence intervals and significance
        coef_items = [(k,v) for k,v in fixed_effects['coef'].items() if k != 'Intercept']
        feature_names = [x[0] for x in coef_items]
        coefs = [x[1] for x in coef_items]
        
        plt.figure(figsize=(9,6))
        
        # Plot 95% confidence intervals
        plt.errorbar(feature_names, coefs,
                    yerr=[(fixed_effects.loc[name,'coef'] - fixed_effects.loc[name,'ci_lower']) for name in feature_names],
                    fmt='none', color='black', capsize=5)
        
        bars = plt.bar(feature_names, coefs)
        
        # Add significance markers based on corrected p-values
        for idx, name in enumerate(feature_names):
            if fixed_effects.loc[name,'qvalue'] < 0.001:
                plt.text(idx, coefs[idx], '***', ha='center', va='bottom', fontsize=18)
            elif fixed_effects.loc[name,'qvalue'] < 0.01:
                plt.text(idx, coefs[idx], '**', ha='center', va='bottom', fontsize=18)
            elif fixed_effects.loc[name,'qvalue'] < 0.05:
                plt.text(idx, coefs[idx], '*', ha='center', va='bottom', fontsize=18)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        plt.xticks(rotation=45, ha='right', fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(-0.25,0.25)
        plt.title(f'Fixed Effects Coefficients for {df.roi.iloc[0]}', fontsize=22)
        plt.xlabel('Features', fontsize=22)
        plt.ylabel('Beta', fontsize=22)
        plt.tight_layout()
        plt.show()
        
    return results_dict

def analyze_remarkable_correlations(group_correlation_df, banned_candidate_dict, candidate_rsm_dict, neural_rsm, r_threshold=0.1):
    """Analyze regression coefficients for ROIs with significant correlations
    Args:
        group_correlation_df (pd.DataFrame): DataFrame with columns [reference, candidate, module, r, p, q], as the criterion to select candidates of remarkable correlations with neural RSMs
        banned_candidate_dict (dict): Dictionary of banned candidates for each module
        candidate_rsm_dict (dict): Dictionary of candidate RSMs
        neural_rsm (dict): Dictionary of neural RSMs
        r_threshold (float): Correlation threshold for including ROIs
    Returns:
        dict: Dictionary of regression coefficients for each ROI
    """
    df = group_correlation_df.copy()
    sig_rows = df[np.abs(df.r) >= r_threshold]
    # Get union of all candidates across ROIs and exclude banned candidates
    candidate_list = {}
    total_comparisons = 0
    for module in candidate_rsm_dict.keys():
        module_rows = sig_rows[sig_rows['module'] == module]
        if not module_rows.empty:
            candidates = list(set(module_rows['candidate'].tolist()))
            # Remove banned candidates for this module
            if module in banned_candidate_dict.keys():
                candidates = [c for c in candidates if c not in banned_candidate_dict[module]]
            candidate_list[module] = candidates
            total_comparisons += len(candidates)
    
    roi_groups = sig_rows['reference'].unique()
    total_comparisons *= len(roi_groups)
    
    # run linear mixed-effects model
    results_lm = {}
    #results_lr = {}
    for roi in roi_groups:
        print("Linear mixed-effects model for ",roi)
        # Prepare regression data using the same candidate list for all ROIs
        reg_df = prepare_regression_data(roi, candidate_list, candidate_rsm_dict, neural_rsm)
        
        #results_lr[roi] = fit_regression_model(reg_df)
        results_lm[roi] = fit_mixed_model(reg_df, multiple_comparison='bonferroni', n_total_comparison=total_comparisons)
        
    return results_lr, results_lm

# %%
# Filter for FFA_l reference only
results_lr, results_lm = analyze_remarkable_correlations(average_df, {"CLIP_annotation":[],"model_embedding":["resnet3d50","multi_resnet3d50","CLIP"]}, combined_rsm, roi_neural_rsm, r_threshold=0.1)

# %%

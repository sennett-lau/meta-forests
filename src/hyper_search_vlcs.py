import itertools
import pandas as pd
import os
import numpy as np
import time
from main import meta_forests_on_vlcs, vlcs_load_and_extract_features


def meta_forests_hyperparameter_search_on_vlcs():
    """
    Perform hyperparameter search for the MetaForests model on VLCS dataset.
    """
    # Define hyperparameter grid
    param_grid = {
        'epochs': [10, 30],
        'alpha': [-0.5, -2.0],
        'beta': [0.5, 2.0],
        'epsilon': [1e-7, 1e-10],
        'per_random_forest_n_estimators': [50, 150],
        'per_random_forest_max_depth': [3, 5, 7]
    }
    
    # Fixed parameters
    fixed_params = {
        'random_state': 42,
        'baseline_random_state': 42
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))
    
    # Create parameter dictionaries for each combination
    all_params = []
    for i, combo in enumerate(combinations):
        param_dict = {k: v for k, v in zip(keys, combo)}
        param_dict.update(fixed_params)
        param_dict['run_id'] = i
        all_params.append(param_dict)
    
    print(f"Total number of combinations to test: {len(all_params)}")
    
    # Initialize or load results dataframe
    results_file = "src/results/vlcs_hyperparameter_search_results.csv"
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        print(f"Loaded existing results file with {len(results_df)} entries")
    else:
        columns = list(keys) + list(fixed_params.keys()) + ['run_id', 'meta_forests_accuracy', 
                                                          'baseline_accuracy', 'improvement', 'runtime']
        results_df = pd.DataFrame(columns=columns)
        print("Created new results dataframe")
    
    # Keep track of best performance
    best_improvement = -float('inf')
    best_params = None

    vlcs_domains, training_extracted_features, testing_extracted_features = vlcs_load_and_extract_features()
    
    # Run through all combinations
    for params in all_params:
        run_id = params['run_id']
        
        # Check if this combination has already been tested
        if not results_df.empty and run_id in results_df['run_id'].values:
            print(f"Skipping run_id {run_id} as it was already tested")
            
            # Update best improvement if this was better
            run_results = results_df[results_df['run_id'] == run_id].iloc[0]
            if run_results['improvement'] > best_improvement:
                best_improvement = run_results['improvement']
                best_params = {k: run_results[k] for k in keys}
                
            continue
        
        print(f"\n----- Running hyperparameter combination {run_id+1}/{len(all_params)} -----")
        for k, v in params.items():
            if k != 'run_id':
                print(f"{k}: {v}")
        
        # Start timing
        start_time = time.time()
        
        # Train and evaluate model with this parameter combination
        try:
            meta_forests_accuracy, baseline_accuracy, improvement = meta_forests_on_vlcs(
                epochs=params['epochs'],
                alpha=params['alpha'],
                beta=params['beta'],
                epsilon=params['epsilon'],
                random_state=params['random_state'],
                baseline_random_state=params['baseline_random_state'],
                per_random_forest_n_estimators=params['per_random_forest_n_estimators'],
                per_random_forest_max_depth=params['per_random_forest_max_depth'],
                vlcs_domains=vlcs_domains,
                training_extracted_features=training_extracted_features,
                testing_extracted_features=testing_extracted_features
            )
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Save results
            result_row = {**params, 
                         'meta_forests_accuracy': meta_forests_accuracy, 
                         'baseline_accuracy': baseline_accuracy,
                         'improvement': improvement,
                         'runtime': runtime}
            
            # Create a new DataFrame with the same columns as results_df to avoid the warning
            new_row_df = pd.DataFrame([result_row], columns=results_df.columns)
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            results_df.to_csv(results_file, index=False)
            
            # Update best performance
            if improvement > best_improvement:
                best_improvement = improvement
                best_params = {k: params[k] for k in keys}
                
            print(f"Completed in {runtime:.2f} seconds with improvement: {improvement:.4f}")
            
        except Exception as e:
            print(f"Error running combination {run_id}: {str(e)}")
            # Log the error in the results
            result_row = {**params, 
                         'meta_forests_accuracy': np.nan, 
                         'baseline_accuracy': np.nan,
                         'improvement': np.nan,
                         'runtime': time.time() - start_time,
                         'error': str(e)}
            
            # Create a new DataFrame with the same columns as results_df to avoid the warning
            new_row_df = pd.DataFrame([result_row], columns=results_df.columns)
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            results_df.to_csv(results_file, index=False)
    
    # Print best results
    print("\n----- Best Hyperparameters -----")
    print(f"Best improvement: {best_improvement:.4f}")
    for k, v in best_params.items():
        print(f"{k}: {v}")
    
    # Get the best row
    if not results_df.empty and 'improvement' in results_df.columns:
        best_row = results_df.iloc[results_df['improvement'].idxmax()]
        print("\n----- Best Overall Results -----")
        print(f"Meta-Forests accuracy: {best_row['meta_forests_accuracy']:.4f}")
        print(f"Baseline accuracy: {best_row['baseline_accuracy']:.4f}")
        print(f"Improvement: {best_row['improvement']:.4f}")
    
    return best_params, best_improvement

meta_forests_hyperparameter_search_on_vlcs()

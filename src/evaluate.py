import numpy as np
import json
import os

def find_best_model(model_names, metrics_dir, priority_order, best_model_file):
	#Create dictionary to store metrics in priority order for each model
	comparison_dict = {model_name: [] for model_name in model_names}
	#Iterate through metrics in priority order
	for metric_name in priority_order:
		#Add metric to each model
		for model_name in model_names:
			#Loading in metrics file
			model_metrics = json.load(open(os.path.join(metrics_dir, \
				model_name + '_train_results.json')))
			#Saving metrics
			metric = model_metrics["metrics"][-1][metric_name]
			#Accounting for nonsensical results
			if metric < 0 or np.isnan(metric):
				metric = 0
			#Adding it to dictionary
			comparison_dict[model_name].append(metric)
	#Converting each list to tuple
	comparison_dict = {model_name:tuple(metrics) for model_name, metrics \
	in comparison_dict.items()}
	#Returning the key with the highest metrics in priority order
	best_model_name = max(comparison_dict, key=comparison_dict.get)
	with open(best_model_file, 'w') as outfile:
		outfile.write(best_model_name)
	return best_model_name

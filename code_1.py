import pandas as pd
import pyddm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np

#--------------Data Handling----------------
class DataLoader:
    """
    handles loading and filtering of data
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.original_df = None
        self.filtered_df = None

    def load_data(self):
        df = pd.read_csv(self.filepath, sep='\t')
        df['correct'] = df['R'].apply(lambda x: 1 if x == 'positive' else 0)
        self.original_df = df.copy()
        return df

    def filter_outliers(self, df):
        Q1 = df['rt'].quantile(0.25)
        Q3 = df['rt'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_df = df[(df['rt'] >= Q1 - 1.5*IQR) & (df['rt'] <= Q3 + 1.5*IQR)]
        self.filtered_df = filtered_df
        return filtered_df
    
#--------------Data Visualisation----------------
class DataVisualizer:
    """
    handles data visualization
    plot_comparison: plots comparison of correct and error reaction times
    plot_density: plots density plot of correct and error reaction times
    plot_condition_histograms_and_density: plots correct and error reaction times per condition
    plot_rt_per_subject_per_condition: plots average reaction time per subject per condition
    plot_rt_distributions: plots distribution of correct and error reaction times
    """
    def __init__(self, original_df, filtered_df):
        self.original_df = original_df
        self.filtered_df = filtered_df

    def plot_comparison(self):
        original_correct_rts = self.original_df[self.original_df['correct'] == 1]['rt']
        original_error_rts = self.original_df[self.original_df['correct'] == 0]['rt']
        filtered_correct_rts = self.filtered_df[self.filtered_df['correct'] == 1]['rt']
        filtered_error_rts = self.filtered_df[self.filtered_df['correct'] == 0]['rt']

        plt.figure(figsize=(14, 6))

        # Correct reaction times comparison
        plt.subplot(1, 2, 1)
        plt.hist(original_correct_rts, bins=50, alpha=0.5, label='Original Correct RTs', color='blue')
        plt.hist(filtered_correct_rts, bins=50, alpha=0.5, label='IQR-Filtered Correct RTs', color='pink')
        plt.title("Comparison of Correct Reaction Times")
        plt.xlabel("Reaction Time (s)")
        plt.ylabel("Frequency")
        plt.legend()

        # Error reaction times comparison
        plt.subplot(1, 2, 2)
        plt.hist(original_error_rts, bins=50, alpha=0.5, label='Original Error RTs', color='red')
        plt.hist(filtered_error_rts, bins=50, alpha=0.5, label='IQR-Filtered Error RTs', color='orange')
        plt.title("Comparison of Error Reaction Times")
        plt.xlabel("Reaction Time (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig('reaction_time_comparison.png')
        plt.show()

    def plot_density(self):
        plt.figure(figsize=(12, 6))
        sns.kdeplot(self.filtered_df[self.filtered_df['correct'] == 1]['rt'], color='blue', label='Correct RTs', fill=False)
        sns.kdeplot(self.filtered_df[self.filtered_df['correct'] == 0]['rt'], color='orange', label='Error RTs', fill=False)
        plt.title("Density Plot of Reaction Times: Correct vs Error")
        plt.xlabel("Reaction Time (s)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig('density_correct_error.png')
        plt.show()

    def plot_condition_histograms_and_density(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.histplot(self.filtered_df[self.filtered_df['S'] == 'positive']['rt'], bins=50, color='blue', label='Compatible', alpha=0.5, ax=axes[0])
        sns.histplot(self.filtered_df[self.filtered_df['S'] == 'negative']['rt'], bins=50, color='red', label='Incompatible', alpha=0.5, ax=axes[0])
        axes[0].set_title("Histogram of Reaction Times")
        axes[0].set_xlabel("Reaction Time (s)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        
        sns.kdeplot(self.filtered_df[self.filtered_df['S'] == 'positive']['rt'], color='blue', label='Compatible', fill=False, ax=axes[1])
        sns.kdeplot(self.filtered_df[self.filtered_df['S'] == 'negative']['rt'], color='red', label='Incompatible', fill=False, ax=axes[1])
        axes[1].set_title("Density Plot of Reaction Times")
        axes[1].set_xlabel("Reaction Time (s)")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig('reaction_times_conditions.png')
        plt.show()  

    def plot_rt_per_subject_per_condition(self):
        # Average reaction time per subject, condition, and correctness
        avg_rt = self.filtered_df.groupby(['subjects', 'S', 'correct'])['rt'].mean().reset_index()

        # Create separate data for compatible and incompatible conditions
        compatible = avg_rt[avg_rt['S'] == 'positive']
        incompatible = avg_rt[avg_rt['S'] == 'negative']

        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(data=compatible, x='subjects', y='rt', hue='correct', palette='muted')
        plt.title("Average Reaction Time per Subject (Compatible)")
        plt.xlabel("Subject ID")
        plt.ylabel("Average Reaction Time (s)")
        plt.legend(title="Correctness", labels=["Incorrect", "Correct"])
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        sns.barplot(data=incompatible, x='subjects', y='rt', hue='correct', palette='muted')
        plt.title("Average Reaction Time per Subject (Incompatible)")
        plt.xlabel("Subject ID")
        plt.ylabel("Average Reaction Time (s)")
        plt.legend(title="Correctness", labels=["Incorrect", "Correct"])
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('avg_rt_per_subject_per_condition.png')
        plt.show()

#--------------DDM Model----------------
class DDMModeler:
    """
    handles fitting of DDM model
    create_sample: creates a pyddm sample from the filtered data
    define_model: defines the DDM model
    fit_model: fits the model to the sample
    extract_fitted_parameters: extracts the fitted parameters from the model
    """

    def __init__(self, df):
        self.df = df
        self.sample = None
        self.model = None

    def create_sample(self):
        self.sample = pyddm.Sample.from_pandas_dataframe(
            self.df, rt_column_name='rt', correct_column_name='correct'
        )
        return self.sample

    def define_model(self):
        self.model = pyddm.Model(
            drift=pyddm.DriftConstant(drift=pyddm.Fittable(minval=-2, maxval=2, default=.5)),
            noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=.1, maxval=1, default=.8)),
            bound=pyddm.BoundConstant(B=pyddm.Fittable(minval=.3, maxval=1.5, default=1.2)),
            overlay=pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=.5, default=.3)),
            T_dur=4,
            dt=.001
        )
        return self.model

    def fit_model(self):
        pyddm.fit_adjust_model(
            model=self.model, 
            sample=self.sample, 
            lossfunction=pyddm.LossRobustLikelihood, 
            verbose=True
        )
        # print(self.model.get_dependence("drift"))
        # print(dir(self.model.get_dependence("drift")))

        pyddm.display_model(self.model)

    def extract_fitted_parameters(self):
        # Access components through get_dependence()
        params = {
            'drift': self.model.get_dependence("drift").drift,
            'noise': self.model.get_dependence("noise").noise,
            'B': self.model.get_dependence("bound").B,
            'nondectime': self.model.get_dependence("overlay").nondectime,
        }
        return params
    
    def plot_rt_distributions(self):
        correct_rts = self.sample.corr
        error_rts = self.sample.err

        plt.figure(figsize=(8, 6))
        ax1 = plt.subplot(2, 1, 1)
        plt.hist(correct_rts, bins=50, alpha=0.75, label="Correct Responses")
        plt.title("Correct Reaction Time Distribution")
        plt.xlabel("Reaction Time (s)")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(2, 1, 2, sharey=ax1)
        plt.hist(error_rts, bins=50, alpha=0.75, label="Error Responses", color='orange')
        plt.title("Error Reaction Time Distribution")
        plt.xlabel("Reaction Time (s)")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.savefig('reaction_time_distribution.png')
        plt.show()

#--------------Parameter Analysis----------------
class ParameterExtractor:
    """
    handles extraction of fitted parameters
    fit_model_per_subject_condition: fits the DDM model for each subject and condition
    save_results: saves the fitted parameters to a CSV file
    """
    def __init__(self, filtered_df):
        self.filtered_df = filtered_df
        self.results = []

    def fit_model_per_subject_condition(self):
        for subject in self.filtered_df['subjects'].unique():
            for condition in self.filtered_df['S'].unique():
                subset = self.filtered_df[(self.filtered_df['subjects'] == subject) & (self.filtered_df['S'] == condition)]
                if subset.empty:
                    continue

                ddm_modeler = DDMModeler(subset)
                ddm_modeler.create_sample()
                ddm_modeler.define_model()
                ddm_modeler.fit_model()
                params = ddm_modeler.extract_fitted_parameters()
                params.update({'ID': subject, 'Condition': condition})	 
                self.results.append(params)

    def save_results(self, filename='fitted_parameters.csv'):
        results_df = pd.DataFrame(self.results)
        column_order = ['ID', 'Condition', 'drift', 'noise', 'B', 'nondectime']
        results_df = results_df[column_order]
        results_df.to_csv(filename, index=False)
        print(f"Reordered parameter table saved to {filename}")
        return results_df

#--------------DDM Visualization----------------
class DDMParameterVisualizer:
    """
    handles visualization of DDM parameters
    plot_parameters_by_condition: plots separate lines for each participant for each parameter by condition
    """
    def __init__(self, parameters_df):
        self.parameters_df = parameters_df

    def plot_parameters_by_condition(self):
        parameters_to_plot = ['drift', 'noise', 'B', 'nondectime']

        for parameter in parameters_to_plot:
            plt.figure(figsize=(8, 6))
            sns.lineplot(
                data=self.parameters_df,
                x='Condition',
                y=parameter,
                hue='ID',
                marker='o',
                palette='tab10'
            )
            plt.title(f'{parameter.capitalize()} Across Conditions')
            plt.xlabel('Condition')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend(title='Participant ID', bbox_to_anchor=(1.05, 1), loc='upper left') 
            plt.tight_layout()
            plt.savefig(f'{parameter}_by_condition.png')
            plt.show()

    def paired_t_tests(self, output_filename='t_test_results.csv'):
        parameters_to_test = ['drift', 'noise', 'B', 'nondectime']
        results = []
        for parameter in parameters_to_test:
            compatible_values = self.parameters_df[self.parameters_df['Condition'] == 'positive'][parameter]
            incompatible_values = self.parameters_df[self.parameters_df['Condition'] == 'negative'][parameter]
            t_stat, p_value = ttest_rel(compatible_values, incompatible_values)
            results.append({
                'Parameter': parameter,
                'T-Statistic': t_stat,
                'P-Value': p_value
            })
            

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_filename, index=False)
        print(f"T-test results saved to {output_filename}")

        return results_df        
    
#--------------Psychometric and Chronomentric Model----------------
class FunctionVisualizer:
    """
    handles visualization of psychometric and chronometric functions
    plot_functions: plots the psychometric and chronometric functions
    """
    def __init__(self, sample, model):
        self.sample = sample
        self.model = model

    def plot_functions(self):
        coherences = sorted(self.sample.condition_values('S'))  # Replace 'S' with your actual condition name
        psychometric_model = [self.model.solve(conditions={"S": coh}).prob("correct") for coh in coherences]
        psychometric_sample = [self.sample.subset(S=coh).prob("correct") for coh in coherences]
        chronometric_model = [self.model.solve(conditions={"S": coh}).mean_decision_time() for coh in coherences]
        chronometric_sample = [self.sample.subset(S=coh).mean_decision_time() for coh in coherences]

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(coherences, psychometric_sample, label="Sample")
        plt.plot(coherences, psychometric_model, label="Model")
        plt.title("Psychometric Function")
        plt.xlabel("Coherence")
        plt.ylabel("P(correct)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(coherences, chronometric_sample, label="Sample")
        plt.plot(coherences, chronometric_model, label="Model")
        plt.title("Chronometric Function")
        plt.xlabel("Coherence")
        plt.ylabel("Mean RT")
        plt.legend()

        plt.tight_layout()
        plt.savefig('psychometric_chronometric_functions.png')
        plt.show()

#--------------Summary Table----------------
class SummaryTableGenerator:
    """
    handles generation and saving of summary tables
    generate_and_save_summary: generates and saves summary tables for compatible and incompatible conditions
    perform_paired_t_test: performs a paired t-test on the reaction times
    """
    def __init__(self, filtered_df):
        self.filtered_df = filtered_df

    def generate_and_save_summary(self, filename_compatible='summary_table_compatible.csv', filename_incompatible='summary_table_incompatible.csv'):
        self.filtered_df['S'] = self.filtered_df['S'].replace({
            'positive': 'Compatible',
            'negative': 'Incompatible'
        })

        compatible_table = self.filtered_df[self.filtered_df['S'] == 'Compatible'].groupby(['subjects', 'S']).agg(median_reaction_time=('rt', 'median')).reset_index()
        compatible_table.rename(columns={
            'subjects': 'ID',
            'S': 'Condition',
            'median_reaction_time': 'RT'
        }, inplace=True)

        compatible_table.to_csv(filename_compatible, index=False)
        print(f"Compatible summary table saved to {filename_compatible}")
        incompatible_table = self.filtered_df[self.filtered_df['S'] == 'Incompatible'].groupby(['subjects', 'S']).agg(median_reaction_time=('rt', 'median')).reset_index()
        incompatible_table.rename(columns={
            'subjects': 'ID',
            'S': 'Condition',
            'median_reaction_time': 'RT'
        }, inplace=True)

        incompatible_table.to_csv(filename_incompatible, index=False)
        print(f"Incompatible summary table saved to {filename_incompatible}")
        return compatible_table, incompatible_table
    
    def perform_paired_t_test(self):
        compatible_table = self.filtered_df[self.filtered_df['S'] == 'Compatible'].groupby('subjects').agg(
            median_reaction_time=('rt', 'median')
        ).reset_index()

        incompatible_table = self.filtered_df[self.filtered_df['S'] == 'Incompatible'].groupby('subjects').agg(
            median_reaction_time=('rt', 'median')
        ).reset_index()
        merged = pd.merge(compatible_table, incompatible_table, on='subjects', suffixes=('_compatible', '_incompatible'))
        t_stat, p_value = ttest_rel(merged['median_reaction_time_compatible'], merged['median_reaction_time_incompatible'])

        print(f"Paired t-Test Results:\nT-Statistic: {t_stat:.3f}, P-Value: {p_value:.3f}")
        return t_stat, p_value

#--------------T-Test Visualizer----------------
class TTestVisualizer:
    """
    handles visualization of t-test results
    plot_ttest_results: plots the t-test results
    """
    def __init__(self, summary_compatible, summary_incompatible):
        self.summary_compatible = summary_compatible
        self.summary_incompatible = summary_incompatible

    def plot_ttest_results(self):
        mean_rt_compatible = self.summary_compatible['RT'].mean()
        mean_rt_incompatible = self.summary_incompatible['RT'].mean()
        ttest_rt = ttest_rel(self.summary_compatible['RT'], self.summary_incompatible['RT'])

        se_rt_compatible = self.summary_compatible['RT'].std() / np.sqrt(len(self.summary_compatible))
        se_rt_incompatible = self.summary_incompatible['RT'].std() / np.sqrt(len(self.summary_incompatible))

        plt.figure(figsize=(10, 6))
        plt.bar(['Compatible', 'Incompatible'], [mean_rt_compatible, mean_rt_incompatible],
                yerr=[se_rt_compatible, se_rt_incompatible], color=['blue', 'red'], alpha=0.7, capsize=5)
        plt.title('Paired T-Test of Reaction Times per Condition')
        plt.ylabel('Mean Reaction Time (s)')
        plt.annotate(f"T: {ttest_rt.statistic:.3f}, p: {ttest_rt.pvalue:.3f}",
                xy=(0.5, max(mean_rt_compatible, mean_rt_incompatible)),
                ha='center', va='bottom', fontsize=10, color='black')
        plt.tight_layout()
        plt.savefig('ttest_reaction_times_with_errorbars.png')
        plt.show()

#--------------Main----------------
def main():
    #1. Load and preprocess the data
    data_loader = DataLoader('dataset-16.tsv')
    data = data_loader.load_data()
    filtered_data = data_loader.filter_outliers(data)

    #2. Visualize reaction times
    rt_visualizer = DataVisualizer(data_loader.original_df, filtered_data)
    rt_visualizer.plot_comparison()
    rt_visualizer.plot_density()
    rt_visualizer.plot_condition_histograms_and_density()
    rt_visualizer.plot_rt_per_subject_per_condition()

    #3. T-test and summary table
    summary_generator = SummaryTableGenerator(filtered_data)
    summary_compatible, summary_incompatible = summary_generator.generate_and_save_summary()
    t_stat, p_value = summary_generator.perform_paired_t_test()

    visualizer = TTestVisualizer(summary_compatible, summary_incompatible)
    visualizer.plot_ttest_results()

    #4. DDM Model
    ddm_modeler = DDMModeler(filtered_data)
    sample = ddm_modeler.create_sample()
    model = ddm_modeler.define_model()
    ddm_modeler.fit_model()
    ddm_modeler.plot_rt_distributions()

    #5. Visualize functions
    function_visualizer = FunctionVisualizer(sample, model)
    function_visualizer.plot_functions()

    #6. Extract and visualize parameters
    parameter_extractor = ParameterExtractor(filtered_data)
    parameter_extractor.fit_model_per_subject_condition()
    fitted_parameters = parameter_extractor.save_results()
    fitted_parameters_df = pd.read_csv('fitted_parameters.csv')
    visualizer = DDMParameterVisualizer(fitted_parameters_df)
    visualizer.plot_parameters_by_condition()
    t_test_results = visualizer.paired_t_tests('t_test_results.csv')

if __name__ == "__main__":
    main()


    


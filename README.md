<div>
  <img src="Github_Readme_Graphic.png" alt="Project Banner" style="border: 2px solid #0366d6; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);" width="100%">
</div>

# Deep Learning Charity Success Predictor

## Overview
This project leverages deep learning techniques to develop a sophisticated binary classification model for Alphabet Soup Nonprofit. The primary objective is to predict the success 
probability of organizations seeking funding, enabling more informed decision-making in grant allocation.

Key Features:
- Neural network architecture optimized for binary classification
- Analysis of 34,000+ historical funding cases
- Comprehensive feature engineering and preprocessing
- Model performance metrics and optimization steps
- Actionable insights for funding decisions

The complete implementation, including data preprocessing, model training, and evaluation, can be found in [@AlphabetSoupCharity_Funding.ipynb](AlphabetSoupCharity_Funding.ipynb).

## Data Preprocessing

## Original Dataset Columns
As shown in [@AlphabetSoupCharity_Funding.ipynb](AlphabetSoupCharity_Funding.ipynb), the dataset contains the following columns:

* **EIN** (Employer Identification Number)
  - Unique identifier for each organization
  - 9-digit number assigned by IRS
  - Removed as a feature as it's purely for identification

* **NAME**
  - Organization's legal name
  - Text field containing business names
  - Removed as it's an identification field, not predictive (added & binned in the optimization model)

* **APPLICATION_TYPE**
  - Type of application submitted
  - 17 unique categories (T3, T4, T5, etc.)
  - Indicates purpose/nature of funding request
  - Binned to reduce rare categories

* **AFFILIATION**
  - Organization's affiliation type
  - Categories include Independent, CompanySponsored, etc.
  - Indicates relationship with other entities
  - Used as-is in model

* **CLASSIFICATION**
  - Government classification of organization
  - 71 unique codes (C1000, C2000, etc.)
  - Represents organization's primary activity/purpose
  - Binned to consolidate rare classifications

* **USE_CASE**
  - Intended use of requested funding
  - Categories like Healthcare, Education, etc.
  - Direct indicator of project purpose
  - Used as-is in initial model

* **ORGANIZATION**
  - Organization structure type
  - Categories: Association, Trust, Corporation, etc.
  - Legal/operational structure indicator
  - Used as-is initially, later evaluated for importance

* **STATUS**
  - Active status of organization
  - Binary indicator (1 = Active, 0 = Inactive)
  - Basic operational status metric
  - Later dropped due to low correlation with success

* **INCOME_AMT**
  - Income classification of organization
  - Ranges from 0 to 50M+
  - Categorical ranges of annual income
  - Used as-is, some rare categories later consolidated

* **SPECIAL_CONSIDERATIONS**
  - Flag for special considerations
  - Binary indicator (Y/N)
  - Marks applications needing extra review
  - Later dropped due to low predictive value

* **ASK_AMT**
  - Funding amount requested
  - Numerical value in dollars
  - Direct measure of project scale
  - Used as-is in model

* **IS_SUCCESSFUL**
  - Target variable
  - Binary outcome (1 = Successful, 0 = Unsuccessful)
  - Indicates if funding was used effectively
  - Based on Alphabet Soup's success criteria


### Initial Data Model Processing
As implemented in [@AlphabetSoupCharity_Funding.ipynb](AlphabetSoupCharity_Funding.ipynb):

* **Target Variable**: `IS_SUCCESSFUL` (Binary outcome)
* **Features Used**: 
  - APPLICATION_TYPE (Binned)
  - AFFILIATION
  - CLASSIFICATION (Binned)
  - USE_CASE
  - ORGANIZATION
  - STATUS
  - INCOME_AMT
  - SPECIAL_CONSIDERATIONS
  - ASK_AMT
* **Removed Features**: 'EIN', 'NAME' (Identification columns)

### Feature Binning Strategy
From [@AlphabetSoupCharity_Funding.ipynb](AlphabetSoupCharity_Funding.ipynb):

* **APPLICATION_TYPE Binning**:
  - Analyzed value counts of APPLICATION_TYPE
  - Found 17 unique application types with varying frequencies
  - Applications with counts < 500 were grouped into "Other"
  - Reduced from 17 to 9 distinct categories
  - Rationale: Consolidating rare application types reduces noise while preserving major patterns
  - This threshold balanced information retention with model complexity

* **CLASSIFICATION Binning**:
  - Examined distribution of CLASSIFICATION values
  - Original data contained 71 unique classifications
  - Classifications with counts < 1000 were consolidated as "Other" 
  - Reduced from 71 to 6 distinct categories
  - Rationale: 
    * Many classifications had very few examples (<100 occurrences)
    * Grouping rare classes improves model generalization
    * Prevents overfitting to rare categories
    * Maintains statistical significance of major classification groups


### Optimization Model Preprocessing
Building upon [@AlphabetSoupCharity_Optimization.ipynb](AlphabetSoupCharity_Optimization.ipynb):

1. **Feature Reduction**:

   * Dropped 12 less influential columns based on feature importance analysis:
     - SPECIAL_CONSIDERATIONS_Y: Low variance binary feature with minimal predictive value
     - AFFILIATION_Regional: Less common affiliation type representing a small subset
     - CLASSIFICATION_C7000: Classification category with lower frequency occurrence
     - USE_CASE_Other: Generic catch-all category providing limited distinctive information
     - INCOME_AMT_50M+: Very rare income bracket with few samples
     - APPLICATION_TYPE_T7: Application type with lower frequency and impact
     - ORGANIZATION_Association: Redundant as organization info captured by AFFILIATION
     - STATUS: Binary feature showing low correlation with target variable
     - CLASSIFICATION_C1700: Low frequency classification with only 287 occurrences
     - APPLICATION_TYPE_T8: Lower impact application type (737 occurrences)
     - INCOME_AMT_1-9999: Small income bracket that can be merged with next tier
     - USE_CASE_Preservation: Less distinctive use case with minimal predictive power

2. **Updated Features Used**:
   * After feature reduction and optimization, the final feature set included:
     - NAME (Binned for frequent organizations)
     - APPLICATION_TYPE (Binned, remaining categories)
     - CLASSIFICATION (Binned, remaining categories)
     - ORGANIZATION (Key categories)
     - INCOME_AMT (Selected brackets)
     - ASK_AMT
   * Key changes from initial feature set:
     - Removed low-impact features identified in feature importance analysis
     - Consolidated categorical variables through strategic binning
     - Retained features showing strong predictive power
     - Introduced binned NAME feature for additional signal
   * Rationale:
     - Focus on features with demonstrated predictive value
     - Reduce model complexity while maintaining performance
     - Balance between information retention and model efficiency


3. **Name Column Binning**:
   * Analyzed frequency distribution of organization names
   * Applied binning strategy for names appearing less than 100 times:
     - Names occurring < 100 times were grouped into "Other" category
     - Preserved high-frequency organization names
   * This reduced noise from rare organization names while maintaining important patterns from frequently occurring organizations
   * Results showed improved model stability and reduced overfitting by consolidating sparse categorical data

3. **Scaling Implementation**:
   The scaling implementation involved using StandardScaler to normalize the feature data:
   1. First, a StandardScaler instance was created to standardize features by removing the mean and scaling to unit variance
   2. The scaler was fit to the training data (X_train_reduced) to learn the mean and standard deviation
   3. The fitted scaler was then used to transform both training and test datasets, ensuring consistent scaling across all data
   
   This standardization step is crucial for neural networks as it helps achieve faster convergence during training and prevents features with larger magnitudes 
   from dominating the model's learning process.

## Model Development

### Initial Model
As implemented in [@AlphabetSoupCharity_Funding.ipynb](AlphabetSoupCharity_Funding.ipynb):
* **Architecture**:
  - Input features: 43
  - First hidden layer: 80 neurons (ReLU)
  - Second hidden layer: 30 neurons (ReLU)
  - Output layer: 1 neuron (Sigmoid)
* **Performance**:
  - Loss: 0.5578
  - Accuracy: 72.63%

### Optimized Model
Building upon [@AlphabetSoupCharity_Optimization.ipynb](AlphabetSoupCharity_Optimization.ipynb):
* **Architecture**:
  - Input features: 68 (after feature reduction)
  - First hidden layer: 10 neurons (ReLU)
  - Second hidden layer: 8 neurons (Sigmoid)
  - Third hidden layer: 6 neurons (Sigmoid)
  - Output layer: 1 neuron (Sigmoid)
* **Training Parameters**:
  - Optimizer: Adam
  - Loss: Binary crossentropy
  - Metrics: Accuracy
  - Epochs: 100

## Performance Analysis

### Model Comparison

The optimized model showed improved performance compared to the initial model in [@AlphabetSoupCharity_Funding.ipynb](AlphabetSoupCharity_Funding.ipynb):

 | Metric   | Initial Model | Optimized Model | Improvement |
 |----------|---------------|-----------------|-------------|
 | Loss     | 0.5601       | 0.4942         | -12.2%      |
 | Accuracy | 72.89%       | 75.38%         | +2.61%      |

The optimization efforts resulted in:
- A reduction in loss from 0.5601 to 0.4942 (12.2% improvement)
- An increase in accuracy from 72.89% to 75.38% (2.61 percentage point gain)
- More stable training performance with the additional hidden layer
- Better generalization through the modified architecture and neuron configuration

The improvements made in the optimized model demonstrate that the architectural changes and hyperparameter tuning successfully enhanced the model's 
predictive capabilities while reducing overfitting. The significant reduction in loss coupled with improved accuracy validates the effectiveness of our optimization approach.


## Contact

For questions, suggestions, or collaboration opportunities regarding this deep learning project, please contact:

* **Author**: [Sergei N. Sergeev]
* **Email**: [sergei.sergeev.n@gmail.com]
* **LinkedIn**: [Sergei Sergeev](https://www.linkedin.com/in/sergei-sergeev-4a607269/)
* **GitHub**: [Sergei Sergeev](https://github.com/LackOfWitness)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows for reuse of this code with minimal restrictions. It includes permissions for:
- Commercial use
- Modification
- Distribution
- Private use

While maintaining limited liability and offering no warranty. Full license terms are available in the LICENSE file.



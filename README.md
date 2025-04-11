Land Use and Cultivation Analysis (2015–2016)
This project explores the Estimated Area by Size Class and Land Use dataset, standardized by NDAP (National Data & Analytics Platform), to analyze patterns in agricultural land utilization across India for the year 2015–2016.

📊 Project Overview
The goal of this project is to perform Exploratory Data Analysis (EDA) and basic linear regression to identify relationships and trends among land-use types—such as net area sown, net area cultivated, uncultivated land, and fallows—based on landholding categories and Indian states.

🧠 Key Objectives
Analyze distributions and averages for key features like net area sown/cultivated.
Visualize landholding category distribution across the dataset.
Understand state-wise agricultural area differences.
Detect correlations between land-use features.
Perform linear regression to predict net area cultivated from net area sown.
Evaluate model performance using Mean Squared Error (MSE).

📂 Dataset Description
Title: Estimated Area By Size Class And Land Use (Standardised By NDAP)
Collected by: Department of Agriculture, Cooperation and Farmers Welfare, Ministry of Agriculture & Farmers Welfare
Time period: 2015–2016
Granularity: Sub-district level
Coverage: All agricultural lands operated as one technical unit, regardless of legal title
Key attributes:

Category of holdings
Net area sown
Area under current fallows
Net area cultivated
Uncultivated area
State and Subdistrict names

📈 Visualizations & Insights
Histogram: Net Area Sown distribution across holdings
Bar Plot: Average Net Area Cultivated per Holding Category
Pie Chart: Landholding Category Distribution
Top 10 States: With highest average Net Area Sown
Correlation Heatmap: Between major land use attributes
Scatter Plot + Regression Line: Net Area Sown vs Net Area Cultivated
Horizontal Bar Charts: Top-performing states by area attributes

🔍 Regression Results
Model: Simple Linear Regression
Input: Net area sown
Target: Net area cultivated
MSE: 3,229,024.2922
Although the MSE appears large, the values in the dataset are in large area units (hectares), making this error more interpretable in context.

🛠️ Tech Stack
Python
Pandas
Seaborn
Matplotlib
Scikit-learn

📌 How to Use
Clone this repository
git clone https://github.com/rachit-448/agri-land-insights.git

Install dependencies
pip install -r requirements.txt

Run the Python file
python EDA_Project.py

📝 License
This project is licensed under the MIT License.
Feel free to use, modify, or contribute!

🔗 Acknowledgments
Dataset from ndap.niti.gov.in

NDAP (National Data & Analytics Platform)

Ministry of Agriculture & Farmers Welfare

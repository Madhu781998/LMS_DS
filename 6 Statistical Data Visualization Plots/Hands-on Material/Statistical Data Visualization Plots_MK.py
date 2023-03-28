import pandas as pd
import matplotlib.pyplot as plt

#1(a)
# Read data into Python
data = pd.read_csv("C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/6 Statistical Data Visualization Plots/Hands-on Material/Statistical Datasets/Q1_a.csv")

# Third moment business decision
data.speed.skew()
data.dist.skew() 

# Fourth moment business decision
data.speed.kurt()
data.dist.kurt()

#Histogram
plt.hist(data.speed)
plt.hist(data.dist)

# Measures of Central Tendency / First moment business decision
data.speed.mean()
data.speed.median()
data.speed.mode()

# Measures of Dispersion / Second moment business decision
data.speed.var() # variance
data.speed.std() # standard deviation
range = max(data.speed) - min(data.speed) # range
range

#1(b)
import pandas as pd

dt = pd.read_csv("C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/6 Statistical Data Visualization Plots/Hands-on Material/Statistical Datasets/Q2_b.csv")

# Third moment business decision
dt.SP.skew()
dt.WT.skew() 

# Fourth moment business decision
dt.SP.kurt()
dt.WT.kurt()


#Q(3)
import pandas as pd

dt1 = pd.read_excel("C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/6 Statistical Data Visualization Plots/Hands-on Material/Q3.xlsx")
dt1.Marks.mean()
dt1.Marks.median()
dt1.Marks.var()
dt1.Marks.std()


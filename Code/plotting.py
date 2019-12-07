import numpy as np
import matplotlib.pyplot as plt

#Create values and labels for bar chart
values =np.random.rand(3)
inds   =np.arange(3)
labels = ["A","B","C"]

#Plot a bar chart
plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.bar(inds, values, align='center') #This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Error") #Y-axis label
plt.xlabel("Method") #X-axis label
plt.title("Error vs Method") #Plot title
plt.xlim(-0.5,2.5) #set x axis range
plt.ylim(0,1) #Set yaxis range

#Set the bar labels
plt.gca().set_xticks(inds) #label locations
plt.gca().set_xticklabels(labels) #label values

#Save the chart
plt.savefig("../Figures/example_bar_chart.pdf")

#Create values and labels for line graphs
values =np.random.rand(2,5)
inds   =np.arange(5)
labels =["Method A","Method B"]

#Plot a line graph
plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(inds,values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Error") #Y-axis label
plt.xlabel("Value") #X-axis label
plt.title("Error vs Value") #Plot title
plt.xlim(-0.1,4.1) #set x axis range
plt.ylim(0,1) #Set yaxis range
plt.legend(labels,loc="best")

#Save the chart
plt.savefig("../Figures/example_line_plot.pdf")

#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()

#Displays the charts.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()


plt.subplot(421)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, pred_y, 'ro' ,linestyle = 'None')
plt.subplot(425)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, predy_trig, 'ro' ,linestyle = 'None')

plt.subplot(422)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, pred_y, 'ro' ,linestyle = 'None')
plt.subplot(426)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, predy_trig, 'ro' ,linestyle = 'None')
degree = 6 
trig_degree = 10
pred_y, predy_trig = KRR_scratch(train_x, train_y, test_x, degree, trig_degree, lmbda, delta)
pred_y, predy_trig = BERR_scratch(train_x, train_y, test_x, degree, trig_degree, lmbda, delta)

plt.subplot(423)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, pred_y, 'ro' ,linestyle = 'None')
plt.subplot(427)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, predy_trig, 'ro' ,linestyle = 'None')

plt.subplot(424)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, pred_y, 'ro' ,linestyle = 'None')
plt.subplot(428)
plt.plot(test_x, test_y, 'b*' ,linestyle = 'None')
plt.plot(test_x, predy_trig, 'ro' ,linestyle = 'None')





try:
    import pandas as pd
    from scipy import stats
    import seaborn as sns
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ttest_ind
except:
    print('Some Packages are Missing')


def clean_data(DF1, DF2, N):
    """This function cleans the Data Frames to be processed by removing 
    the rows containing more than N zero columns
    
    Parameters
    ----------
    DF1 : DataFrame
        This is the first data frame file to be processed

    DF2 : DataFrame
        This is the second data frame file to be processed

    N : int
        The threshold number of zeros to drop the row from both
        data frames

    Returns
    -------
    This function works inplace, means that it works on the original
    data frames and does not return anything
    """

    number_of_zeros = 0                                                         # Create a variable to hold the number of zeros per row.
    rows_to_drop = set()                                                        # Holds the index of the rows in both data frames to be dropped in a set (non-repeating indexes).
    for row in range(len(DF1)):                                                 # Loops over the rows of the first data frame.
        for column in range(2, len(DF1.columns)):                               # then iterating over each value.
            if DF1.iloc[row, column] == 0:                                      # if that specific location is equal to zero
                number_of_zeros += 1                                            # increament the number_of_zeros by one
            if number_of_zeros >= N:                                            # and then checks wheather the number of zeros exceeds the threshold
                rows_to_drop.add(row)                                           # if yes then we add that row index to the set of rows to be dropped
                break                                                           # and we stop iterating for that row
        number_of_zeros = 0                                                     # after each row operation, we reset the number_of_zeros
        
    for row in range(len(DF2)):                                                 # we do the same thing for the second data frame
        for column in range(2,len(DF2.columns)):
            if DF2.iloc[row, column] == 0:
                number_of_zeros += 1
            if number_of_zeros >= N:
                rows_to_drop.add(row)
                break
        number_of_zeros = 0

    indexes = list(rows_to_drop)                                                # Then we Cast the set of indexes into a list

    DF1.drop(indexes, inplace=True)                                             # we now drop the rows from both the data frames
    DF2.drop(indexes, inplace=True)


def get_pearson(DF1, DF2):
    """Creates the Pearson Correlation Coefficient for each one of the genes in the two data frames

    Parameters
    ----------
    DF1 : DataFrame
        The first data frame

    DF2 : DataFrame
        The second data frame

    Returns
    -------
    correlation : dict
        a Key value pair containing the name of each gene with its correlation coefficient

    indexes : dict
        a key value pair containing the index of each gene with its correlation coefficient
    """

    correlation = {}                                                            # A dictionary that holds the key as the gene name and the value as the correlation coefficient of that gene
    indexes = {}                                                                # A dictionary that holds the key as the index of the gene and the value as the correlation coefficient of that gene
    for row in range(len(DF1)):                                                 # iterate over the rows of the data frames
        G_1 = DF1.iloc[row, 2:]                                                 # this holds the value of the columns in the first data frame
        G_2 = DF2.iloc[row, 2:]                                                 # this holds the value of the columns in the second data frame
        r, _ = stats.pearsonr(G_1, G_2)                                         # now we call the method pearsonr from the scipy package to get the correlation coefficient
        correlation[DF1.iloc[row, 0]] = r                                       # this holds the correlation coefficient of each gene name in the correlation dict
        indexes[row] = r                                                        # this holds the correlation coefficient of each index name in the indexes dict
    return correlation, indexes                                                 # the method return both the correlation and the indexes

def get_max(correlation, indexes):
    """Gets the Max Correlation Coefficient with its index and gene name

    Parameters
    ----------
    correlation: dict
        contains the names of the genes with their correlation coefficient

    indexes : dict
        contains the indexes of the genes with their correlation coefficient
    
    Returns
    -------
    max_key: str
        The name of the gene with maximun correlation coefficient

    max_value: list[int]
        The value of the maximum correlation coefficient

    max_index: list[int]
        The index of the maximum correlation coefficient in the data frame
    """
    max_value = max(correlation.values())                                           # get the maximum value of the correlation coefficient
    max_keys = [k for k, v in correlation.items() if v == max_value]                # getting all genes containing the maximum value
    max_index = [k for k, v in indexes.items() if v == max_value]                   # getting all indexes of the genes containing the maximum value

    return max_keys, max_value, max_index


def get_min(correlation, indexes):
    """
    Gets the Min Correlation Coefficient with its index and gene name

    Parameters
    ----------
    correlation: dict
        contains the names of the genes with their correlation coefficient
    indexes
        contains the indexes of the genes with their correlation coefficient
    
    Returns
    -------
    min_key: String
        The name of the gene with minimun correlation coefficient
    min_value: list[int]
        The value of the minimum correlation coefficient
    min_index: list[int]
        The index of the minimum correlation coefficient in the data frame
    """
    min_value = min(correlation.values())                                           # get the minimum value of the correlation coefficient
    min_keys = [k for k, v in correlation.items() if v == min_value]                # getting all genes containing the minimum value
    min_index = [k for k, v in indexes.items() if v == min_value]                   # getting all indexes of the genes containing the minimum value

    return min_value, min_keys, min_index


def scatter_plot(DF1, DF2, index, xlabel=None, ylabel=None, legend=None, title=None):
    """
    Grapghs the scatter plot for one row from the two data frames

    Parameters
    ----------
    DF1 : DataFrame
        the first data frame (x-axis)

    DF2 : DataFrame
        the second data frame (y-axis)

    index : int
        the index of the row to be scattered

    xlable : str, optional
        the label to be placed on the x-axis

    ylable : str, optional
        the label to be placed on the y-axis

    legend : str, optional
        the label of the scattered points
    
    Returns
    -------
    the method does not return anything, just shows the scatter plot on the screen
    """

    plt.style.use('seaborn-bright')
    x = DF1.iloc[index, 2:]                                                         # The x-axis of the graph is the row from the first data frame
    y = DF2.iloc[index, 2:]                                                         # The y-axis of the graph is the row from the second data frame
    plt.scatter(x, y, c='b', marker='.')
    x = x.astype('float64')
    y = y.astype('float64')
    sns.regplot(x, y)                                                               # Graph the Scatter Plot with the regression Line
    if(xlabel):                                                                     # if the end-user entered an x-label it will be added here
        plt.xlabel(xlabel)
    if(ylabel):                                                                     # the same goes for the y-label
        plt.ylabel(ylabel)
    plt.scatter(x, y)
    if(legend):                                                                     # and the legend of the scatter plot
        plt.legend([legend])
    if(title):
        plt.title(title)
    plt.grid()

    plt.show()


def p_values_paired(DF1, DF2, alpha):
    """Gets the P-Value of the two paired data frames

    Parameters
    ----------
    DF1 : DataFrame.
        The first Data frame

    DF1 : DataFrame.
        The second Data frame

    alpha : float.
        The Significance Level

    Returns
    -------
    diffrentially_genes : DataFrame.
        The filtered genes with corrected p-value less than alpha
    """
    
    # create a list that holds all of the genes names
    gene_names = []
    # create a list that holds all of the genes p-value
    p_values = []
    # iterate over the rows of the two data frames
    for row in range(len(DF1)):
        # hold the row data of each data frame in a seperate variable
        G_1 = DF1.iloc[row, 2:]
        G_2 = DF2.iloc[row, 2:]
        # append the name of that gene to the gene names list
        gene_names.append(DF1.iloc[row, 0])
        # calculate the p value for the two paired rows
        p_val = ttest_rel(G_1, G_2).pvalue
        # append the value of that p to the p-value list
        p_values.append(p_val)
    # calculate the corrected p-value by the fdr method
    corrected_p_values = multipletests(p_values, alpha=alpha, method='fdr_bh')[1]
    # create a data frame that contains the gene names, their p-value, and their corrected p-value.
    significance_genes = pd.DataFrame({'Gene_name':gene_names, 'p-values':p_values, 'p-values_fdr':corrected_p_values})
    # append a new column to the data frame with a boolean value that checks wheather the p-value is less than the alpha or not
    significance_genes['significance:p_vlaue'] = significance_genes['p-values'].apply(lambda x: x < alpha)
    # do the same thing for the corrected p-value
    significance_genes['significance:p_vlaue_fdr'] = significance_genes['p-values_fdr'].apply(lambda x: x < alpha)
    # create a second data frame that holds only those genes with a corrected p-value less than the alpha
    diffrentially_genes = significance_genes[significance_genes['significance:p_vlaue_fdr']== True]
    # return that data frame
    return diffrentially_genes

def p_values_ind(DF1, DF2, alpha):
    """Gets the P-Value of the two independant data frames

    Parameters
    ----------
    DF1 : DataFrame.
        The first Data frame

    DF1 : DataFrame.
        The second Data frame

    alpha : float.
        The Significance Level

    Returns
    -------
    diffrentially_genes : DataFrame.
        The filtered genes with corrected p-value less than alpha
    """

    # create a list that holds all of the genes names
    gene_names = []
    # create a list that holds all of the genes p-value
    p_values = []
    # iterate over the rows of the two data frames
    for row in range(len(DF1)):
        # hold the row data of each data frame in a seperate variable
        G_1 = DF1.iloc[row, 2:]
        G_2 = DF2.iloc[row, 2:]
        # append the name of that gene to the gene names list
        gene_names.append(DF1.iloc[row, 0])
        # calculate the p-value for the two independant rows
        p_val = ttest_ind(G_1, G_2).pvalue
        # append the value of that p to the p-value list
        p_values.append(p_val)
    # calculate the corrected p-value by the fdr method
    corrected_p_values = multipletests(p_values, alpha=alpha, method='fdr_bh')[1]
    # create a data frame that contains the gene names, their p-value, and their corrected p-value.
    significance_genes = pd.DataFrame({'Gene_name':gene_names, 'p-values':p_values, 'p-values_fdr':corrected_p_values})
    # append a new column to the data frame with a boolean value that checks wheather the p-value is less than the alpha or not
    significance_genes['significance:p_vlaue'] = significance_genes['p-values'].apply(lambda x: x < alpha)
    # do the same thing for the corrected p-value
    significance_genes['significance:p_vlaue_fdr'] = significance_genes['p-values_fdr'].apply(lambda x: x < alpha)
    # create a second data frame that holds only those genes with a corrected p-value less than the alpha
    diffrentially_genes = significance_genes[significance_genes['significance:p_vlaue_fdr']== True]
    # return that data frame
    return diffrentially_genes

def compare(diffrentially_genes_paired, diffrentially_genes_ind):
    """Get the Common and distinct Genes between two data frames

    Parameters
    ----------
    diffrentially_genes_paired : DataFrame.
        Contains the first data frame with its genes names and fdr corrected paired p-values

    diffrentially_genes_ind : DataFrame.
        Contains the second data frame with its genes names and fdr corrected independant p-values

    Returns
    -------
    common : DataFrames.
        the common genes between the two data frames

    paired_only : DataFrames.
        the genes in the paired data frame and not in the independant one.

    ind_only : DataFrames.
        the genes in the independant data frame and not in the paired one.
    """
    genes_paired = set()
    genes_ind = set()

    for row in range(len(diffrentially_genes_paired)):
        genes_paired.add(diffrentially_genes_paired.iloc[row, 0])
    for row in range(len(diffrentially_genes_ind)):
        genes_ind.add(diffrentially_genes_ind.iloc[row, 0])

    common = genes_paired.intersection(genes_ind)
    ind_only = genes_ind.difference(genes_paired)
    paired_only = genes_paired.difference(genes_ind)

    a = list(common)
    common = diffrentially_genes_paired[diffrentially_genes_paired['Gene_name'].isin(a)]

    b = list(paired_only)
    paired_only = diffrentially_genes_paired[diffrentially_genes_paired['Gene_name'].isin(b)]

    c = list(ind_only)
    ind_only = diffrentially_genes_ind[diffrentially_genes_ind['Gene_name'].isin(c)]



    return common, paired_only, ind_only
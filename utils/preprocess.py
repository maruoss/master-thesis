import numpy as np


# Binary y label generator.
def binary_categorize(y):
    """
    Input: continuous target variable 

    Output: 1 for positive returns, 
            0 for negative returns
    """
    # threshold 0%
    if y > 0:
        return 1
    else:
        return 0


# Multiclass y label generator.
def multi_categorize(y: float, classes: int):
    """
    Creates categorical labels from continuous values.

        Args:
            y (float):      continuous target variable (option return)
            classes (int):  number of classes to create
        Returns:
            (int):          class assignment
        CAREFUL: classes have to be between [0, C) for F.crossentropyloss.
    """
    if classes == 3:
        # thresholds: +/- 5%
        if y > 0.05:
            return 2
        elif y < -0.05:
            return 0
        else:
            return 1
    elif classes == 5:
        # thresholds: +/- 2.5% and +/- 5%
        if y > 0.05:
            return 4
        elif (y > 0.025 and y <= 0.05):
            return 3
        elif (y >= -0.05 and y < -0.025):
            return 1
        elif (y < -0.05):
            return 0
        else:
            return 2 # all returns \elin [-0.025, 0.025]
    # elif classes==10:
    #     if y > 0.05:
    #         return 9
    #     elif (y > 0.04 and y <= 0.05):
    #         return 8
    #     elif (y > 0.03 and y <= 0.04):
    #         return 7
    #     elif (y > 0.02 and y <= 0.03):
    #         return 6
    #     elif (y > 0.01 and y <= 0.02):
    #         return 5
    #     elif (y >= -0.02 and y < -0.01):
    #         return 3
    #     elif (y >= -0.03 and y < -0.02):
    #         return 2
    #     elif (y >= -0.04 and y < -0.03):
    #         return 1
    #     elif (y >= -0.05 and y < -0.05):
    #         return 0
    #     else:
    #         return 4
    else:
        raise ValueError("Only multi for 3 or 5 classes implemented right now.")


class YearEndIndeces:
    """Generator for indices where years change.

        Args:
            dates (pandas.Series):      series of datetimes,
            init_train_length (int):    initial train length,
            val_length (int):           validation length
    """
    def __init__(self, dates, init_train_length, val_length, test_length):
        # Find indeces where years change.
        self.val_length = val_length
        self.test_length = test_length
        # Get end of month indeces for slicing.
        # TECHNICALLY its start of month indeces, i.e. first row of January 31,
        # but because for slicing [:idx], idx is not included, we name it end of
        # year here.
        self.eoy_idx =  np.where((dates.dt.year.diff() == 1))[0]
        # Append last row as end of year of last year.
        self.eoy_idx = np.append(self.eoy_idx, len(dates))

        assert init_train_length + val_length + test_length <= len(self.eoy_idx), \
            ("defined train and val are larger than eoy_indeces generated")
        assert init_train_length > 0, "init_train_length must be strictly greater than 0"

        # The 4th idx in eoy_idx is the end of year 5. -> Subtract 1.
        self.train_length_zeroindex = init_train_length - 1

        self.train_eoy = self.eoy_idx[self.train_length_zeroindex:-(val_length+test_length)]
        self.val_eoy = self.eoy_idx[self.train_length_zeroindex + val_length:-test_length]
        # For generate_idx():
        self.test_eoy = self.eoy_idx[self.train_length_zeroindex + val_length + test_length:]

    # def generate(self):
    #     for i in range(len(self.eoy_idx) - (self.train_start_idx + self.val_length)):
    #         yield (list(range(self.train_eoy[i])),
    #                list(range(self.train_eoy[i], self.val_eoy[i])))

    def generate_idx(self):
        for i in range(len(self.eoy_idx) - (self.train_length_zeroindex + self.val_length 
                        + self.test_length)):
            yield ({"train": self.train_eoy[i], 
                    "val": self.val_eoy[i], 
                    "test": self.test_eoy[i]}
                )


class YearMonthEndIndeces:
    """Generator for indices where months change.

        Args:
            dates (pandas.Series):      series of datetimes,
            init_train_length (int):    initial train length,
            val_length (int):           validation length
    """
    def __init__(self, dates, init_train_length, val_length, test_length):
        # self.val_length = val_length
        # self.test_length = test_length
        # Get end of month indeces for slicing.
        # TECHNICALLY its start of month indeces, i.e. first row of January 31,
        # but because for slicing [:idx], idx is not included, we name it end of
        # year here.
        self.eom_idx =  np.concatenate([
                        np.where((dates.dt.month.diff() == 1))[0], 
                        np.where((dates.dt.month.diff() == -11))[0] #Dec->Jan
                        ])
        # Sort, since Dec->Jan months indeces are only concatenated at the end.
        self.eom_idx.sort()
        # Append last row as end of month of last month.
        self.eom_idx = np.append(self.eom_idx, len(dates))

        # End of year indeces
        self.eoy_idx =  np.where((dates.dt.year.diff() == 1))[0]
        self.eoy_idx = np.append(self.eoy_idx, len(dates))

        # Careful: -2 because November (-> cao (2021) return calc.) and December 2021 is not in dataset.
        assert (26 * 12 - 2 == len(self.eom_idx)), ("Some end of month indeces are missing.")
        assert init_train_length > 0, "init_train_length must be strictly greater than 0."

        # The 4th idx in eoy_idx is the end of year 5. -> Subtract 1.
        self.train_length_zeroindex = init_train_length - 1

        # Get eoy indeces where we predicted on AND FIRST ENTRY IS EOY_VAL == SOY_TEST
        self.test_eoy = self.eoy_idx[self.train_length_zeroindex + val_length:]

        # Get first end of month idx of year X until first end of month of year Y
        # a prediction was made.
        years_predicted = np.arange(1996 + init_train_length + val_length, 2021 + 1) #upper limit not included.
        self.month_idx_per_year = {}
        for i, eoy_idx in enumerate(self.test_eoy[:-1]): #-1 because [last_index:last_index+13] not needed.
            idx_in_idx = np.where(np.in1d(self.eom_idx, eoy_idx))[0].item() #only one eom_idx equals one eoy_idx
            # + 13, so that slicing is from start of year until +12 months (end of year).
            self.month_idx_per_year[years_predicted[i]] = self.eom_idx[idx_in_idx:idx_in_idx+13]

        # Check that dictionary years are correct and that months are consecutive.
        assert check_month_years(self.month_idx_per_year, dates=dates), ("Years of end "
        "of month indices are wrong or the months are not consecutive.")

    def get_indeces(self):
        # Return Tuple.
        return (self.test_eoy, self.month_idx_per_year)


def check_month_years(dic, dates):
    """Checks whether all end of month indeces in the dictionary 'dic'
    are in the correct year. Also, checks whether all indeces are in
    consecutive order. 31.12.2019[31.01.2020,......,31.12.2020, 31.01.2021]
    
    The last month of the eom indeces overlaps with the first entry in
    the next year.

    ---
    Example:
        If a year has 12 months in the data, the end of month indeces should 
        have length 13. The first index is the first "row" of the year, 
        the last index is the first row of the next year.
    """
    for year in dic.keys():
        len_dic = len(dic[year])
        for idx, eom_idx in enumerate(dic[year]):
            # Special case: last eom_idx is first eom_idx of next year.
            if idx == len_dic - 1: #idx uses zero indexing.
                if int(year) != dates[eom_idx-1].year or (idx)!= dates[eom_idx-1].month:
                    return False
            elif int(year) != dates[eom_idx].year or (idx+1) != dates[eom_idx].month:
                return False
    return True
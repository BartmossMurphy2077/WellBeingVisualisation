# Data Audit Report

## Join Summary
- Join candidate ['geo', 'time'] unique=1.000 overlap=0.000
- No confident join keys found.

## File Inventory
```
                                                                    dataset  rows  cols  duplicates
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time   409     3           0
            ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time  8022     3           0
          ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time  8022     3           0
      ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time  8976     3           0
                 ddf--datapoints--cell_phones_per_100_people--by--geo--time 10350     3           0
                          ddf--datapoints--cell_phones_total--by--geo--time 10374     3           0
      ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time  7449     3           0
    ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time  7449     3           0
             ddf--datapoints--data_quality_income_per_person--by--geo--time 18778     3           0
     ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time  5492     3           0
```

## Dataset: ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time

### Missingness
```
                                                                    dataset                                      column  missing_count  missing_pct
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time                                         geo              0          0.0
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time                                        time              0          0.0
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time alcohol_consumption_per_adult_15plus_litres              0          0.0
```

### Numeric Distributions
```
                                                                    dataset                                      column     mean      std  median  iqr     skew  kurtosis
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time alcohol_consumption_per_adult_15plus_litres 7.053496 4.856419    6.53 7.09 0.453985  -0.38369
```

### Categorical Distributions
```
                                                                    dataset column  cardinality  entropy                             top_values
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time   time           22 1.690474 2005:187;2008:186;1994:6;1990:4;1995:4
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time    geo          188 7.468362        rus:12;swe:11;aut:3;bra:3;dnk:3
```

### Outlier Diagnostics
```
                                                                    dataset                                      column  outlier_pct
ddf--datapoints--alcohol_consumption_per_adult_15plus_litres--by--geo--time alcohol_consumption_per_adult_15plus_litres      0.00489
```

### Potential Targets
```
                                     column  score                            reasons
alcohol_consumption_per_adult_15plus_litres      3 proxy_keyword,variance,scale_0_100
```

## Dataset: ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time

### Missingness
```
                                                        dataset                          column  missing_count  missing_pct
ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time                             geo              0          0.0
ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time                            time              0          0.0
ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time body_mass_index_bmi_men_kgperm2              0          0.0
```

### Numeric Distributions
```
                                                        dataset                          column      mean      std  median  iqr     skew  kurtosis
ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time body_mass_index_bmi_men_kgperm2 23.788245 2.658898    24.0  3.9 0.224145 -0.032842
```

### Categorical Distributions
```
                                                        dataset column  cardinality  entropy                                   top_values
ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time   time           42 5.392317 1975:191;1976:191;1977:191;1978:191;1979:191
ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time    geo          191 7.577429           afg:42;ago:42;alb:42;and:42;are:42
```

### Outlier Diagnostics
```
                                                        dataset                          column  outlier_pct
ddf--datapoints--body_mass_index_bmi_men_kgperm2--by--geo--time body_mass_index_bmi_men_kgperm2     0.007105
```

### Potential Targets
```
                         column  score                            reasons
body_mass_index_bmi_men_kgperm2      3 proxy_keyword,variance,scale_0_100
```

## Dataset: ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time

### Missingness
```
                                                          dataset                            column  missing_count  missing_pct
ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time                               geo              0          0.0
ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time                              time              0          0.0
ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time body_mass_index_bmi_women_kgperm2              0          0.0
```

### Numeric Distributions
```
                                                          dataset                            column      mean    std  median  iqr     skew  kurtosis
ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time body_mass_index_bmi_women_kgperm2 24.535477 2.7913    24.7  3.7 0.180541  0.343236
```

### Categorical Distributions
```
                                                          dataset column  cardinality  entropy                                   top_values
ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time   time           42 5.392317 1975:191;1976:191;1977:191;1978:191;1979:191
ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time    geo          191 7.577429           afg:42;ago:42;alb:42;and:42;are:42
```

### Outlier Diagnostics
```
                                                          dataset                            column  outlier_pct
ddf--datapoints--body_mass_index_bmi_women_kgperm2--by--geo--time body_mass_index_bmi_women_kgperm2     0.013962
```

### Potential Targets
```
                           column  score                            reasons
body_mass_index_bmi_women_kgperm2      3 proxy_keyword,variance,scale_0_100
```

## Dataset: ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time

### Missingness
```
                                                              dataset                                column  missing_count  missing_pct
ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time                                   geo              0          0.0
ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time                                  time              0          0.0
ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time breast_cancer_deaths_per_100000_women              0          0.0
```

### Numeric Distributions
```
                                                              dataset                                column      mean      std  median  iqr     skew  kurtosis
ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time breast_cancer_deaths_per_100000_women 19.102059 8.534001   17.88 10.7 1.162597   2.86337
```

### Categorical Distributions
```
                                                              dataset column  cardinality  entropy                                   top_values
ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time   time           44 5.459432 1980:204;1981:204;1982:204;1983:204;1984:204
ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time    geo          204 7.672425           afg:44;ago:44;alb:44;and:44;are:44
```

### Outlier Diagnostics
```
                                                              dataset                                column  outlier_pct
ddf--datapoints--breast_cancer_deaths_per_100000_women--by--geo--time breast_cancer_deaths_per_100000_women     0.020499
```

### Potential Targets
```
                               column  score                            reasons
breast_cancer_deaths_per_100000_women      3 proxy_keyword,variance,scale_0_100
```

## Dataset: ddf--datapoints--cell_phones_per_100_people--by--geo--time

### Missingness
```
                                                   dataset                     column  missing_count  missing_pct
ddf--datapoints--cell_phones_per_100_people--by--geo--time                        geo              0          0.0
ddf--datapoints--cell_phones_per_100_people--by--geo--time                       time              0          0.0
ddf--datapoints--cell_phones_per_100_people--by--geo--time cell_phones_per_100_people              0          0.0
```

### Numeric Distributions
```
                                                   dataset                     column        mean       std     median       iqr      skew  kurtosis
ddf--datapoints--cell_phones_per_100_people--by--geo--time                       time 1997.140000 15.842934 1998.00000 26.000000 -0.238463 -0.808284
ddf--datapoints--cell_phones_per_100_people--by--geo--time cell_phones_per_100_people   37.917323 52.661927    1.36867 79.201975  1.208306  0.716490
```

### Categorical Distributions
```
                                                   dataset column  cardinality  entropy                         top_values
ddf--datapoints--cell_phones_per_100_people--by--geo--time    geo          212 7.699435 afg:52;ago:52;alb:52;and:52;bgd:52
```

### Outlier Diagnostics
```
                                                   dataset                     column  outlier_pct
ddf--datapoints--cell_phones_per_100_people--by--geo--time                       time     0.000000
ddf--datapoints--cell_phones_per_100_people--by--geo--time cell_phones_per_100_people     0.004831
```

### Potential Targets
```
                    column  score  reasons
cell_phones_per_100_people      1 variance
```

## Dataset: ddf--datapoints--cell_phones_total--by--geo--time

### Missingness
```
                                          dataset            column  missing_count  missing_pct
ddf--datapoints--cell_phones_total--by--geo--time               geo              0          0.0
ddf--datapoints--cell_phones_total--by--geo--time              time              0          0.0
ddf--datapoints--cell_phones_total--by--geo--time cell_phones_total              0          0.0
```

### Numeric Distributions
```
                                          dataset            column         mean          std  median       iqr      skew   kurtosis
ddf--datapoints--cell_phones_total--by--geo--time              time 1.997119e+03 1.583180e+01  1998.0      26.0 -0.235153  -0.807941
ddf--datapoints--cell_phones_total--by--geo--time cell_phones_total 1.225005e+07 7.359449e+07 28950.0 3192502.5 15.589155 294.729760
```

### Categorical Distributions
```
                                          dataset column  cardinality  entropy                         top_values
ddf--datapoints--cell_phones_total--by--geo--time    geo          212  7.70003 afg:52;ago:52;alb:52;and:52;bgd:52
```

### Outlier Diagnostics
```
                                          dataset            column  outlier_pct
ddf--datapoints--cell_phones_total--by--geo--time              time     0.000000
ddf--datapoints--cell_phones_total--by--geo--time cell_phones_total     0.169848
```

### Potential Targets
```
           column  score  reasons
cell_phones_total      1 variance
```

## Dataset: ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time

### Missingness
```
                                                              dataset                                column  missing_count  missing_pct
ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time                                   geo              0          0.0
ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time                                  time              0          0.0
ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time cholesterol_fat_in_blood_men_mmolperl              0          0.0
```

### Numeric Distributions
```
No data available.
```

### Categorical Distributions
```
                                                              dataset                                column  cardinality  entropy                                   top_values
ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time                                  time           39 5.285402 1980:191;1981:191;1982:191;1983:191;1984:191
ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time cholesterol_fat_in_blood_men_mmolperl           29 4.497795      3.8:584;4.3:567;4.5:536;4.4:496;4.7:452
ddf--datapoints--cholesterol_fat_in_blood_men_mmolperl--by--geo--time                                   geo          191 7.577429           afg:39;ago:39;alb:39;and:39;are:39
```

### Outlier Diagnostics
```
No data available.
```

### Potential Targets
```
                               column  score                           reasons
cholesterol_fat_in_blood_men_mmolperl      3 proxy_keyword,variance,scale_0_10
```

## Dataset: ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time

### Missingness
```
                                                                dataset                                  column  missing_count  missing_pct
ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time                                     geo              0          0.0
ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time                                    time              0          0.0
ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time cholesterol_fat_in_blood_women_mmolperl              0          0.0
```

### Numeric Distributions
```
No data available.
```

### Categorical Distributions
```
                                                                dataset                                  column  cardinality  entropy                                   top_values
ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time                                    time           39 5.285402 1980:191;1981:191;1982:191;1983:191;1984:191
ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time cholesterol_fat_in_blood_women_mmolperl           29 4.389025      4.5:737;4.6:684;4.0:582;4.1:516;4.8:507
ddf--datapoints--cholesterol_fat_in_blood_women_mmolperl--by--geo--time                                     geo          191 7.577429           afg:39;ago:39;alb:39;and:39;are:39
```

### Outlier Diagnostics
```
No data available.
```

### Potential Targets
```
                                 column  score                           reasons
cholesterol_fat_in_blood_women_mmolperl      3 proxy_keyword,variance,scale_0_10
```

## Dataset: ddf--datapoints--data_quality_income_per_person--by--geo--time

### Missingness
```
                                                       dataset                         column  missing_count  missing_pct
ddf--datapoints--data_quality_income_per_person--by--geo--time                            geo              0          0.0
ddf--datapoints--data_quality_income_per_person--by--geo--time                           time              0          0.0
ddf--datapoints--data_quality_income_per_person--by--geo--time data_quality_income_per_person              0          0.0
```

### Numeric Distributions
```
                                                       dataset column        mean      std  median  iqr      skew  kurtosis
ddf--datapoints--data_quality_income_per_person--by--geo--time   time 1966.243902 42.92466  1971.5 41.0 -3.573485 18.118967
```

### Categorical Distributions
```
                                                       dataset                         column  cardinality  entropy                         top_values
ddf--datapoints--data_quality_income_per_person--by--geo--time data_quality_income_per_person            2 0.895424                     0:12920;5:5858
ddf--datapoints--data_quality_income_per_person--by--geo--time                            geo          229 7.839204 abw:82;afg:82;ago:82;aia:82;alb:82
```

### Outlier Diagnostics
```
                                                       dataset column  outlier_pct
ddf--datapoints--data_quality_income_per_person--by--geo--time   time      0.02439
```

### Potential Targets
```
                        column  score             reasons
data_quality_income_per_person      2 variance,scale_0_10
```

## Dataset: ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time

### Missingness
```
                                                               dataset                                 column  missing_count  missing_pct
ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time                                    geo              0          0.0
ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time                                   time              0          0.0
ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time economic_growth_over_the_past_10_years              0          0.0
```

### Numeric Distributions
```
                                                               dataset                                 column    mean      std  median    iqr      skew  kurtosis
ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time economic_growth_over_the_past_10_years 1.76577 2.909446   1.861 3.0145 -0.405107   4.92128
```

### Categorical Distributions
```
                                                               dataset column  cardinality  entropy                                   top_values
ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time   time           45 5.405087 2003:179;2002:173;2001:170;2000:169;1999:159
ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time    geo          180 7.306034           bel:45;aut:45;aus:45;arg:45;cri:45
```

### Outlier Diagnostics
```
                                                               dataset                                 column  outlier_pct
ddf--datapoints--economic_growth_over_the_past_10_years--by--geo--time economic_growth_over_the_past_10_years     0.045339
```

### Potential Targets
```
                                column  score  reasons
economic_growth_over_the_past_10_years      1 variance
```

## Dataset: ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time

### Missingness
```
                                                                     dataset                                       column  missing_count  missing_pct
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time                                          geo              0          0.0
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time                                         time              0          0.0
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time females_aged_15_24_unemployment_rate_percent              0          0.0
```

### Numeric Distributions
```
                                                                     dataset                                       column        mean       std  median    iqr      skew  kurtosis
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time                                         time 2007.009139 12.302329 2009.00 18.000 -0.919805  0.871155
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time females_aged_15_24_unemployment_rate_percent   19.022656 13.495879   15.34 16.698  1.222660  1.262441
```

### Categorical Distributions
```
                                                                     dataset column  cardinality  entropy                         top_values
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time    geo          213 7.111499 usa:77;jpn:57;can:49;esp:48;aus:46
```

### Outlier Diagnostics
```
                                                                     dataset                                       column  outlier_pct
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time                                         time     0.008550
ddf--datapoints--females_aged_15_24_unemployment_rate_percent--by--geo--time females_aged_15_24_unemployment_rate_percent     0.037146
```

### Potential Targets
```
                                      column  score                            reasons
females_aged_15_24_unemployment_rate_percent      3 proxy_keyword,variance,scale_0_100
```

## Dataset: ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time

### Missingness
```
                                                                     dataset                                       column  missing_count  missing_pct
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time                                          geo              0          0.0
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time                                         time              0          0.0
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time females_aged_25_54_unemployment_rate_percent              0          0.0
```

### Numeric Distributions
```
                                                                     dataset                                       column        mean       std   median    iqr      skew  kurtosis
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time                                         time 2007.130603 12.222921 2010.000 17.000 -0.935746  0.958977
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time females_aged_25_54_unemployment_rate_percent    7.581263  6.118198    5.756  6.323  1.834534  4.260003
```

### Categorical Distributions
```
                                                                     dataset column  cardinality  entropy                         top_values
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time    geo          213 7.112446 usa:77;jpn:57;can:49;esp:48;aus:46
```

### Outlier Diagnostics
```
                                                                     dataset                                       column  outlier_pct
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time                                         time     0.010686
ddf--datapoints--females_aged_25_54_unemployment_rate_percent--by--geo--time females_aged_25_54_unemployment_rate_percent     0.054616
```

### Potential Targets
```
                                      column  score                            reasons
females_aged_25_54_unemployment_rate_percent      3 proxy_keyword,variance,scale_0_100
```

## Dataset: ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time

### Missingness
```
                                                                 dataset                                   column  missing_count  missing_pct
ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time                                      geo              0          0.0
ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time                                     time              0          0.0
ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time male_long_term_unemployment_rate_percent              0          0.0
```

### Numeric Distributions
```
                                                                 dataset                                   column     mean     std  median      iqr     skew  kurtosis
ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time male_long_term_unemployment_rate_percent 1.971903 1.92492 1.37355 1.785755 2.183557  7.265297
```

### Categorical Distributions
```
                                                                 dataset column  cardinality  entropy                                 top_values
ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time   time           36 5.014152 2022:102;2019:100;2017:100;2021:95;2018:93
ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time    geo          153 6.694361         aus:35;can:35;bel:35;grc:35;gbr:35
```

### Outlier Diagnostics
```
                                                                 dataset                                   column  outlier_pct
ddf--datapoints--male_long_term_unemployment_rate_percent--by--geo--time male_long_term_unemployment_rate_percent     0.076609
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time suicide_age_15_19_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column     mean      std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time suicide_age_15_19_per_100000_people 6.053169 5.864363     4.2 4.685 2.986379 11.906829
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_15_19_per_100000_people--by--geo--time suicide_age_15_19_per_100000_people     0.065574
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time suicide_age_15_24_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column     mean      std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time suicide_age_15_24_per_100000_people 7.790109 7.676023    5.35 5.945 3.350665 14.855054
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_15_24_per_100000_people--by--geo--time suicide_age_15_24_per_100000_people     0.054645
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time suicide_age_15_29_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column     mean      std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time suicide_age_15_29_per_100000_people 8.756175 8.540141    6.14 6.565 3.257644 13.661122
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_15_29_per_100000_people--by--geo--time suicide_age_15_29_per_100000_people     0.065574
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time suicide_age_25_34_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column      mean       std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time suicide_age_25_34_per_100000_people 10.802459 11.036354     7.6 7.305 3.358848 14.601589
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_25_34_per_100000_people--by--geo--time suicide_age_25_34_per_100000_people     0.081967
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time suicide_age_30_49_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column      mean       std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time suicide_age_30_49_per_100000_people 12.653169 14.866431    10.0 7.805 6.611554 61.833441
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_30_49_per_100000_people--by--geo--time suicide_age_30_49_per_100000_people     0.087432
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time suicide_age_35_44_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column      mean       std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time suicide_age_35_44_per_100000_people 12.899781 16.576807   10.05 7.485 7.186972 69.967712
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_35_44_per_100000_people--by--geo--time suicide_age_35_44_per_100000_people     0.103825
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time suicide_age_45_54_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column     mean       std  median    iqr     skew  kurtosis
ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time suicide_age_45_54_per_100000_people 15.45918 17.890847    11.9 11.965 6.850546 65.936748
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_45_54_per_100000_people--by--geo--time suicide_age_45_54_per_100000_people     0.038251
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time suicide_age_55_64_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column      mean      std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time suicide_age_55_64_per_100000_people 17.558142 14.97421   14.72 17.42 3.018709 17.901351
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_55_64_per_100000_people--by--geo--time suicide_age_55_64_per_100000_people     0.021858
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time suicide_age_65_74_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column      mean       std  median   iqr     skew  kurtosis
ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time suicide_age_65_74_per_100000_people 20.893989 17.185629    15.7 21.88 1.411039  2.767835
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_65_74_per_100000_people--by--geo--time suicide_age_65_74_per_100000_people     0.016393
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time                                 geo              0          0.0
ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time                                time              0          0.0
ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time suicide_age_75_84_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column     mean       std  median    iqr    skew  kurtosis
ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time suicide_age_75_84_per_100000_people 27.47224 22.691131   19.07 30.975 1.01791  0.189611
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--suicide_age_75_84_per_100000_people--by--geo--time suicide_age_75_84_per_100000_people     0.016393
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time

### Missingness
```
                                                             dataset                               column  missing_count  missing_pct
ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time                                  geo              0          0.0
ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time                                 time              0          0.0
ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time suicide_age_85plus_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                             dataset                               column      mean       std  median    iqr     skew  kurtosis
ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time suicide_age_85plus_per_100000_people 59.322022 63.124049   28.89 79.535 1.330251  0.691911
```

### Categorical Distributions
```
                                                             dataset column  cardinality  entropy                    top_values
ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time   time            1  -0.0000                      2019:183
ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time    geo          183   7.5157 afg:1;ago:1;alb:1;are:1;arg:1
```

### Outlier Diagnostics
```
                                                             dataset                               column  outlier_pct
ddf--datapoints--suicide_age_85plus_per_100000_people--by--geo--time suicide_age_85plus_per_100000_people     0.021858
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_men_per_100000_people--by--geo--time

### Missingness
```
                                                      dataset                        column  missing_count  missing_pct
ddf--datapoints--suicide_men_per_100000_people--by--geo--time                           geo              0          0.0
ddf--datapoints--suicide_men_per_100000_people--by--geo--time                          time              0          0.0
ddf--datapoints--suicide_men_per_100000_people--by--geo--time suicide_men_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                      dataset                        column      mean       std   median      iqr     skew  kurtosis
ddf--datapoints--suicide_men_per_100000_people--by--geo--time suicide_men_per_100000_people 14.562286 12.190374 11.00362 12.77937 2.010225  5.894132
```

### Categorical Distributions
```
                                                      dataset column  cardinality  entropy                                   top_values
ddf--datapoints--suicide_men_per_100000_people--by--geo--time   time           22 4.459432 2000:185;2001:185;2002:185;2003:185;2004:185
ddf--datapoints--suicide_men_per_100000_people--by--geo--time    geo          185 7.531381           afg:22;ago:22;alb:22;are:22;arg:22
```

### Outlier Diagnostics
```
                                                      dataset                        column  outlier_pct
ddf--datapoints--suicide_men_per_100000_people--by--geo--time suicide_men_per_100000_people     0.047912
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_per_100000_people--by--geo--time

### Missingness
```
                                                  dataset                    column  missing_count  missing_pct
ddf--datapoints--suicide_per_100000_people--by--geo--time                       geo              0          0.0
ddf--datapoints--suicide_per_100000_people--by--geo--time                      time              0          0.0
ddf--datapoints--suicide_per_100000_people--by--geo--time suicide_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                  dataset                    column     mean      std  median      iqr     skew  kurtosis
ddf--datapoints--suicide_per_100000_people--by--geo--time suicide_per_100000_people 9.474175 7.310143 7.37882 8.301695 1.661949  3.781393
```

### Categorical Distributions
```
                                                  dataset column  cardinality  entropy                                   top_values
ddf--datapoints--suicide_per_100000_people--by--geo--time   time           22 4.459432 2000:185;2001:185;2002:185;2003:185;2004:185
ddf--datapoints--suicide_per_100000_people--by--geo--time    geo          185 7.531381           afg:22;ago:22;alb:22;are:22;arg:22
```

### Outlier Diagnostics
```
                                                  dataset                    column  outlier_pct
ddf--datapoints--suicide_per_100000_people--by--geo--time suicide_per_100000_people      0.04226
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_total_deaths--by--geo--time

### Missingness
```
                                             dataset               column  missing_count  missing_pct
ddf--datapoints--suicide_total_deaths--by--geo--time                  geo              0          0.0
ddf--datapoints--suicide_total_deaths--by--geo--time                 time              0          0.0
ddf--datapoints--suicide_total_deaths--by--geo--time suicide_total_deaths              0          0.0
```

### Numeric Distributions
```
                                             dataset               column        mean          std  median      iqr     skew   kurtosis
ddf--datapoints--suicide_total_deaths--by--geo--time suicide_total_deaths 3560.158436 17427.685422  498.49 1666.695 9.684093 102.619488
```

### Categorical Distributions
```
                                             dataset column  cardinality  entropy                                   top_values
ddf--datapoints--suicide_total_deaths--by--geo--time   time           44 5.459432 1980:204;1981:204;1982:204;1983:204;1984:204
ddf--datapoints--suicide_total_deaths--by--geo--time    geo          204 7.672425           afg:44;ago:44;alb:44;and:44;are:44
```

### Outlier Diagnostics
```
                                             dataset               column  outlier_pct
ddf--datapoints--suicide_total_deaths--by--geo--time suicide_total_deaths     0.115419
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--suicide_women_per_100000_people--by--geo--time

### Missingness
```
                                                        dataset                          column  missing_count  missing_pct
ddf--datapoints--suicide_women_per_100000_people--by--geo--time                             geo              0          0.0
ddf--datapoints--suicide_women_per_100000_people--by--geo--time                            time              0          0.0
ddf--datapoints--suicide_women_per_100000_people--by--geo--time suicide_women_per_100000_people              0          0.0
```

### Numeric Distributions
```
                                                        dataset                          column     mean      std  median      iqr     skew  kurtosis
ddf--datapoints--suicide_women_per_100000_people--by--geo--time suicide_women_per_100000_people 4.560012 3.424574 3.63494 4.293375 1.307407  1.941573
```

### Categorical Distributions
```
                                                        dataset column  cardinality  entropy                                   top_values
ddf--datapoints--suicide_women_per_100000_people--by--geo--time   time           22 4.459432 2000:185;2001:185;2002:185;2003:185;2004:185
ddf--datapoints--suicide_women_per_100000_people--by--geo--time    geo          185 7.531381           afg:22;ago:22;alb:22;are:22;arg:22
```

### Outlier Diagnostics
```
                                                        dataset                          column  outlier_pct
ddf--datapoints--suicide_women_per_100000_people--by--geo--time suicide_women_per_100000_people     0.032187
```

### Potential Targets
```
No data available.
```

## Dataset: ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time

### Missingness
```
                                                            dataset                              column  missing_count  missing_pct
ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time                                 geo              0          0.0
ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time                                time              0          0.0
ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time total_number_of_dollar_billionaires              0          0.0
```

### Numeric Distributions
```
                                                            dataset                              column      mean      std  median  iqr     skew  kurtosis
ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time total_number_of_dollar_billionaires 28.295578 95.62543     5.0 15.0 7.419356 62.722626
```

### Categorical Distributions
```
                                                            dataset column  cardinality  entropy                              top_values
ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time   time           29 4.744019 2024:81;2022:79;2023:79;2025:79;2021:77
ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time    geo           90 6.262562      deu:29;che:29;swe:29;usa:29;sgp:29
```

### Outlier Diagnostics
```
                                                            dataset                              column  outlier_pct
ddf--datapoints--total_number_of_dollar_billionaires--by--geo--time total_number_of_dollar_billionaires     0.141732
```

### Potential Targets
```
                             column  score  reasons
total_number_of_dollar_billionaires      1 variance
```

## Dataset: ddf--datapoints--working_hours_per_week--by--geo--time

### Missingness
```
                                               dataset                 column  missing_count  missing_pct
ddf--datapoints--working_hours_per_week--by--geo--time                    geo              0          0.0
ddf--datapoints--working_hours_per_week--by--geo--time                   time              0          0.0
ddf--datapoints--working_hours_per_week--by--geo--time working_hours_per_week              0          0.0
```

### Numeric Distributions
```
                                               dataset                 column        mean       std  median    iqr      skew  kurtosis
ddf--datapoints--working_hours_per_week--by--geo--time                   time 2009.272401 10.852464 2012.00 15.000 -0.970586  0.715541
ddf--datapoints--working_hours_per_week--by--geo--time working_hours_per_week   39.964640  4.208047   39.41  5.485  0.316560  0.488837
```

### Categorical Distributions
```
                                               dataset column  cardinality  entropy                         top_values
ddf--datapoints--working_hours_per_week--by--geo--time    geo          179  6.85707 can:49;swe:47;usa:46;mlt:45;isr:44
```

### Outlier Diagnostics
```
                                               dataset                 column  outlier_pct
ddf--datapoints--working_hours_per_week--by--geo--time                   time     0.021107
ddf--datapoints--working_hours_per_week--by--geo--time working_hours_per_week     0.010354
```

### Potential Targets
```
                column  score                            reasons
working_hours_per_week      3 proxy_keyword,variance,scale_0_100
```

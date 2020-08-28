# Relax Take-home challenge

The problem statement can be found [here](https://github.com/metacreek/springboard/blob/master/mini-projects/relax_challenge/relax_data_science_challenge.pdf).

The details of the analysis can be found [here](https://github.com/metacreek/springboard/blob/master/mini-projects/relax_challenge/Take%20Home%20%231-functionized.ipynb).

### Most important factor: account creation time

Analysis shows that the most important factor in predicting adoption is the account creation time. We must view this with
some skepticism as older accounts will have the most opportunity to have become adopted at some point over their life; 
similarly, it is possible that newer accounts will eventually become adopted but have not yet had enough time to do so.
See the   
For this reason, I removed this feature and repeated the analysis to determine the next most important factor.

### Second most important factor: organization

The analysis shows that `org_id` is the second most important factor in predicting adoption.  The organization id
appears to be created sequentially, so older organizations would have lower ids.  This could
implicitly mean that this is affected by the same issues that affect account creation time.  For that reason,
I have removed it and repeated the analysis to get the next most important factor.

### Third most important factor: creation source

The account creation source is the next most important predictor in account adoption.  Note the average adoption rate 
from the different creation sources:

| Source | Adoption Rate |
| --- | --: |
| SIGNUP_GOOGLE_AUTH | 16.8 % |
| GUEST_INVITE | 16.6 % |
| SIGNUP | 14.0 % |
| ORG_INVITE | 13.0 % |
| PERSONAL_PROJECTS | 7.8 % |

For the users who were invited to join via guest or organizational invite, we have the
id of the individual who did the invite.  If we restrict our data to just those accounts and repeat the analysis,
we find that the individual who did the invite is more important than the creation source.

 

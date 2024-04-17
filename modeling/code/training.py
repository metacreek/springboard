









X_test.to_pickle('X_test.pickle')


# In[62]:


pd.DataFrame(y_test).to_pickle('y_test.pickle')


# In[63]:


pd.Series(index_test).to_pickle('index_test.pickle')


# In[ ]:


X_train.to_parquet('X_train.parquet')


# In[ ]:


pd.DataFrame(y_train).to_pickle('y_tr')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob


# In[14]:


bashf = open("run_eval.sh","w")
for f in glob("../santanlp-corpus/corpus2/test/*"):
    bashf.write("python run_eval.py --in {} --out results/{}\n".format(f,f[19:]))
bashf.close()


# In[2]:


bashf = open("run_eval2.sh","w")
for f in glob("../santanlp-corpus/corpus3/test/*"):
    bashf.write("python run_eval.py --in {} --out results/{}\n".format(f,f[19:]))
for f in glob("../santanlp-corpus/corpus4/test/*"):
    bashf.write("python run_eval.py --in {} --out results/{}\n".format(f,f[19:]))    
bashf.close()


# In[3]:


bashf = open("run_eval_random.sh","w")
for f in glob("../santanlp-corpus/corpus1/test/*"):
    bashf.write("python run_eval.py --in {} --out results/{}\n".format(f,f[19:]))   
bashf.close()


# In[ ]:





# In[ ]:





# In[ ]:





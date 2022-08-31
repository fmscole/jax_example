I implemented another implementation of CTC algorithm .   
The interface is consistent with pytorch.  
The performance is consistent with optax.ctc_loss alignment.  

just run train.py  




直接运行train.py即可。
本项目基础代码是复制https://github.com/yizt/crnn.pytorch  
copy from https://github.com/yizt/crnn.pytorch  

目的是为了学习ctc_loss。  
在了解了此基础算法之后，用jax实作了一个，通过不断优化，性能最终和官方代码optax.ctc_loss对齐.

（1）与optax.ctc_loss有相同的速度和内存消耗；  
（2）与pytorch的ctcloss有相同的接口；  
（3）更清晰的代码逻辑。   
    
  
有三个主干网络可供选择crnn、cann、carnn。   
通过注释代码，loss可以切换，可以选择官方optax，也可以选择ctcloss_fast 
  

*.txt文件是训练过程，供参考。    
  
由于数据是随机生成，所以无所谓训练集和测试集
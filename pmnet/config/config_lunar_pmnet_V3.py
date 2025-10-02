class config_lunar_pmnet_V3:
    def __init__(self,):
        # basics
        self.batch_size = 16
        self.exp_name = 'config_lunar_pmnet_V3'
        self.num_epochs = 120
        self.start_epoch = 7
        self.val_freq = 1
        self.num_workers = 0
        self.pre_trained_model = '/home/patorrad/Documents/diffusion/StableDiffusion-PyTorch/pmnet/results_k2/config_lunar_pmnet_V3_epoch120/16_0.0001_0.5_20/model_0.00088.pt'
        self.train_ratio = 0.7
        self.validation_ratio = 0.15
        self.test_ratio = 0.15
        self.val_init = 1

        self.dataset_settings()
        self.optim_settings()
        return

    def dataset_settings(self,):
        self.dataset = 'lunar'
        self.cityMap = 'complete'        # complete, height
        self.sampling = 'exclusive' # random, exclusive
        


    def optim_settings(self,):
        self.lr = 1e-4
        self.lr_decay = 0.5
        self.step = 20

    def get_train_parameters(self,):
      return {'exp_name':self.exp_name,
        'batch_size':self.batch_size,
        'num_epochs':self.num_epochs,
        'lr':self.lr,
        'lr_decay':self.lr_decay,
        'step':self.step,
        'sampling':self.sampling}
        
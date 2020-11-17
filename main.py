from dec_module import DEC_Module
###################################################################
# Settings
cfg = {'dec_mode': True, 'train': True, 'test':False, 'evals':False, 'vis': False}

resume = None

if cfg['dec_mode']:
    if cfg['train']:
        dec_model = DEC_Module(multigpu=True, resume=resume)
        dec_model.train(vis=cfg['vis'])
    if cfg['test']:
        dec_model = DEC_Module(multigpu=False, resume=resume)
        dec_model.test()
    if cfg['evals']:
        dec_model = DEC_Module(multigpu=False, resume=resume)
        dec_model.eval_single()

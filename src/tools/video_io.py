from skvideo.io import vread, vwrite
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)

def save_summary(src_video, label, out_video):
    logging.info(f'loading {src_video}')    
    video = vread(src_video)
    summary = video[np.where(label)]
    
    logging.info(f'writing {out_video}')  
    vwrite(out_video,
           summary,
           outputdict={
                  '-vcodec': 'libx264',
                  '-crf': '20',
                  '-pix_fmt': 'yuv420p'
            })
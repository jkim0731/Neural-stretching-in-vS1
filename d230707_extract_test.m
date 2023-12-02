M = [];
M{1} = 'D:\039\plane_6\039_019_000_plane_6.h5';
M{2} = '/data';

config=[];
config = get_defaults(config);
%Set some important settings
config.use_gpu=1;
config.avg_cell_radius=8;
config.trace_output_option='nonneg'; 
config.num_partitions_x=1;
config.num_partitions_y=1;
config.cellfind_min_snr=0.7; 

config.preprocess = 1;
config.compact_output=0;

output2 = extractor(M, config);
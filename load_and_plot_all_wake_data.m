clear;
driver_idx = 0

rng_seeds = [1:40];



root_dir = '/home/elpiloto/Dropbox/code/torch/sleep_eeg_v2/output/sleep_eeg_v2/';
commit_num_and_id = '2_ff5ed02';



for driver_name = {'fullConvWake'}
	driver_idx = driver_idx + 1;
	driver_name = driver_name{:};

	data_dir = fullfile(root_dir, driver_name, commit_num_and_id);

	disp(data_dir)

	rngIdx = 0;
	for rngSeed = rng_seeds
		rngIdx = rngIdx + 1;

		rngStr = ['rng_' num2str(rngSeed)];

		saved_file = fullfile(data_dir, [rngStr '.mat']);

		if ~exist('all_data','var') 
			all_data = load(saved_file);
		else
			all_data( rngIdx ) = load(saved_file);
		end

		figure('Visible','off');
		set(gcf, 'Position', [0 0 2560 1600], 'PaperPositionMode','auto')
		plot(all_data(rngIdx).trainClassAcc); hold all;
		plot(all_data(rngIdx).validClassAcc);
		plot(all_data(rngIdx).trainAvgClassAccSubset);
		plot(all_data(rngIdx).validAvgClassAccSubset);
		xlabel('Training Iteration'); ylabel('Class Accuracy');
		legend({'Train', 'Valid', 'Train - F/P', 'Valid - F/P'});
		
		save_dir = fullfile(data_dir, 'eps');
		if ~exist(save_dir,'dir')
			unix(['mkdir -p ' save_dir ]);
		end
		print(gcf,'-depsc','-painters',fullfile(save_dir,['rng' num2str(rngIdx) '.eps']));
		save_dir = fullfile(data_dir, 'png');
		if ~exist(save_dir,'dir')
			unix(['mkdir -p ' save_dir ]);
		end
		print(gcf,'-dpng','-painters',fullfile(save_dir,['rng' num2str(rngIdx) '.png']));
				
	end

end
close all;

figure(1);
plot(mean([all_data.trainClassAcc],2)); hold all;
plot(mean([all_data.validClassAcc],2));
plot(mean([all_data.trainAvgClassAccSubset],2));
plot(mean([all_data.validAvgClassAccSubset],2));
legend({'Train', 'Valid', 'Train - F/P', 'Valid - F/P'});
xlabel('Training Iteration'); ylabel('Class Accuracy');

save_dir = fullfile(data_dir, 'eps');
print(gcf,'-depsc','-painters',fullfile(save_dir,'avg.eps'));

save_dir = fullfile(data_dir, 'png');
print(gcf,'-dpng','-painters',fullfile(save_dir,'avg.png'));

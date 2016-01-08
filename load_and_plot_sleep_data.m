clear;
driver_idx = 0;

rng_seeds = 1:40;


% TODO: Add automatically grabbing commit num and id
% TODO: Incorporate confusion matrix plots and random accuracy
root_dir = '/home/elpiloto/Dropbox/code/torch/sleep_eeg_v2/output/sleep_eeg_v2/';
commit_num_and_id = '2_ff5ed02';



for driver_name = {'fullConv'}
	driver_idx = driver_idx + 1;
	driver_name = driver_name{:};

	data_dir = fullfile(root_dir, driver_name, commit_num_and_id);

	disp(data_dir)

	rngIdx = 0;
	fileCountIdx = 0;
	for rngSeed = rng_seeds
		rngIdx = rngIdx + 1;

		rngStr = ['rng_' num2str(rngSeed)];

		saved_file = fullfile(data_dir, [rngStr '.mat']);
		if exist(saved_file, 'file')
			fileCountIdx = fileCountIdx + 1;

			if ~exist('all_data','var') 
				all_data = load(saved_file);
			else
				all_data( fileCountIdx ) = load(saved_file);
			end

			figure('Visible','off');
			set(gcf, 'Position', [0 0 2560 1600], 'PaperPositionMode','auto')
			plot(all_data(fileCountIdx).trainClassAcc); hold all;
			plot(all_data(fileCountIdx).validClassAcc);
			plot([0 length(all_data(fileCountIdx).validClassAcc)], [0.5 0.5], '--r');
			xlabel('Training Iteration'); ylabel('Class Accuracy');
			legend({'Train', 'Valid'},'Location','Best');
			
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

end
close all;

figure(1);
plot(mean([all_data.trainClassAcc],2)); hold all;
plot(mean([all_data.validClassAcc],2));
plot([0 length(all_data(fileCountIdx).validClassAcc)], [0.5 0.5], '--r');
legend({'Train', 'Valid'},'Location','Best');
xlabel('Training Iteration'); ylabel('Class Accuracy');

save_dir = fullfile(data_dir, 'eps');
print(gcf,'-depsc','-painters',fullfile(save_dir,'avg.eps'));

save_dir = fullfile(data_dir, 'png');
print(gcf,'-dpng','-painters',fullfile(save_dir,'avg.png'));
close all;

figure(1);
plot(var([all_data.trainClassAcc])); hold all;
plot(var([all_data.validClassAcc]));
legend({'Train', 'Valid'},'Location','Best');
xlabel('Training Iteration'); ylabel('Class Acc. Variance');

save_dir = fullfile(data_dir, 'eps');
print(gcf,'-depsc','-painters',fullfile(save_dir,'std.eps'));

save_dir = fullfile(data_dir, 'png');
print(gcf,'-dpng','-painters',fullfile(save_dir,'std.png'));
close all;

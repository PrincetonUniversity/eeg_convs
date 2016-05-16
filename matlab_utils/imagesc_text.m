function [] = imagesc_text(values, varargin)

    if nargin > 1
        imagesc(values, varargin{:});
    else
        imagesc(values)
    end
    
	x = xlim;
	y = ylim;
	xg = x(1):1:(x(2)-1);
	yg = y(1):1:(y(2)-1);

	[xlbl, ylbl] = meshgrid(xg+0.5, yg+0.5);
	% create cell arrays of number labels
    lbl = arrayfun(@(x) num2str(x,2),values,'UniformOutput',false);
	text(xlbl(:), ylbl(:), lbl(:),'color','k',...
		'HorizontalAlignment','center','VerticalAlignment','middle','FontSize',12);
end

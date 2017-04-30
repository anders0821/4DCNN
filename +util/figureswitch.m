% 参考了opencv的通过标题管理窗口的方法
function [fig] = figureswitch(name)
    % find by name creat if needed
    fids = findall(0,'Type','figure','Name',name);
    if numel(fids)==0
        fig = figure('Name',name,'NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
    else
        fig = fids(1);
    end
    
    % switch to & bring to front
    set(0, 'CurrentFigure', fig);
end

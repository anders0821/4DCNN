% 参考了opencv的通过标题管理窗口的方法
function [] = figureclose(name)
    fids = findall(0,'Type','figure','Name',name);
    for i=1:numel(fids)
        close(fids(i));
    end
end

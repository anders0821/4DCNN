% �ο���opencv��ͨ����������ڵķ���
function [] = figureclose(name)
    fids = findall(0,'Type','figure','Name',name);
    for i=1:numel(fids)
        close(fids(i));
    end
end

classdef GaborParallel < handle
    properties
        pI;
        P;% 固定的P
        O;
        gradP;% 恒为0
        gradO;
    end
    
    methods
        function [this] = GaborParallel(pI, KERNELWIDTH, OUTPUTCHANNELA, OUTPUTCHANNELB)
            assert(mod(KERNELWIDTH, 2)==1);
            % 初始化成员变量
            this.pI = pI;
            [H, W, INPUTCHANNEL, N] = size(this.pI.O);
            this.O = zeros([H W OUTPUTCHANNELA*OUTPUTCHANNELB N]);
            this.gradO = zeros(size(this.O));
            
            % 初始化固定核
            gaborArray = util.gaborFilterBank(OUTPUTCHANNELA,OUTPUTCHANNELB,KERNELWIDTH,KERNELWIDTH);
            gaborArray = cell2mat(gaborArray);
            gaborArray = reshape(gaborArray, [KERNELWIDTH OUTPUTCHANNELA KERNELWIDTH OUTPUTCHANNELB]);
            gaborArray = permute(gaborArray, [1 3 2 4]);
            gaborArray = real(gaborArray);
            this.P = reshape(gaborArray, [KERNELWIDTH KERNELWIDTH 1 OUTPUTCHANNELA*OUTPUTCHANNELB]);
            % figure;
            % for i=1:OUTPUTCHANNELA
            %     for j=1:OUTPUTCHANNELB
            %         subplot(OUTPUTCHANNELA,OUTPUTCHANNELB,(i-1)*OUTPUTCHANNELB+j);
            %         imshow(this.P(:,:,(j-1)*OUTPUTCHANNELA+i), []);
            %     end
            % end
            % drawnow;
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            [H, W, INPUTCHANNEL, N] = size(this.pI.O);
            OUTPUTCHANNEL = size(this.P, 4);
            
            thisO = zeros([H W OUTPUTCHANNEL N]);
            thispIO = this.pI.O;
            thisP = this.P;
            parfor i=1:N
                for j=1:OUTPUTCHANNEL
                    for k=1:INPUTCHANNEL
                        tmp = conv2(thispIO(:,:,k,i), thisP(:,:,k,j), 'same');
                        thisO(:,:,j,i) = thisO(:,:,j,i) + tmp;
                    end
                end
            end
            this.O = thisO;
            
            %toc;
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            % 不做任何处理
            %toc;
            
            this.pI.bp();
        end
        
        function [total, totalP] = debugMemory(this)
            [total, totalP] = this.pI.debugMemory();
            
            disp(util.funname());
            
            fprintf('	O');
            fprintf('	%d', size(this.O));
            fprintf('	(%d)', numel(this.O));
            fprintf('\n');
            
            fprintf('	P');
            fprintf('	%d', size(this.P));
            fprintf('	(%d)', numel(this.P));
            fprintf('\n');
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            fprintf('	gradP');
            fprintf('	%d', size(this.gradP));
            fprintf('	(%d)', numel(this.gradP));
            fprintf('\n');
            
            total = total + numel(this.O)+numel(this.gradO) + numel(this.P)+numel(this.gradP);
            totalP = totalP + numel(this.gradP);
        end
        
        function [p] = debugHist(this, fig1, fig2, fig3, fig4, m, n, p)
            p = this.pI.debugHist(fig1, fig2, fig3, fig4, m, n, p);
            
            p = p+1;
            
            if(~isempty(fig1))
                set(0, 'CurrentFigure', fig1);
                subplot(m,n,p);
                hist(this.O(:),100);
                title(class(this));
            end
            
            if(~isempty(fig2))
                set(0, 'CurrentFigure', fig2);
                subplot(m,n,p);
                hist(this.P(:),100);
                title(class(this));
            end
            
            if(~isempty(fig3))
                set(0, 'CurrentFigure', fig3);
                subplot(m,n,p);
                hist(this.gradO(:),100);
                title(class(this));
            end
            
            if(~isempty(fig4))
                set(0, 'CurrentFigure', fig4);
                subplot(m,n,p);
                hist(this.gradP(:),100);
                title(class(this));
            end
        end
        
        function [Ps] = getPs(this)
            [Ps] = this.pI.getPs();
            Ps{end+1} = this.P;
        end
        
        function [] = setPs(this, Ps)
            this.P = Ps{end};
            Ps = Ps(1:end-1);
            
            this.pI.setPs(Ps);
        end
    end
end

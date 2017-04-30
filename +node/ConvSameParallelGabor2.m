classdef ConvSameParallelGabor2 < handle
    properties
        pI;
        P;
        O;
        gradP;
        gradO;
    end
    
    methods
        function [this] = ConvSameParallelGabor2(pI, KERNELWIDTHS, OUTPUTCHANNELA, OUTPUTCHANNELB)
            assert(all(size(KERNELWIDTHS)==[1 4]));
            assert(all(mod(KERNELWIDTHS, 2)==1));
            assert(KERNELWIDTHS(1)<=size(pI.O,1));
            assert(KERNELWIDTHS(2)<=size(pI.O,2));
            assert(KERNELWIDTHS(3)<=size(pI.O,3));
            assert(KERNELWIDTHS(4)<=size(pI.O,4));
            
            % Gabor核限制为二维核
            assert(KERNELWIDTHS(3)==1);
            assert(KERNELWIDTHS(4)==1);
            
            % 初始化成员变量
            this.pI = pI;
            [H, W, D, T, INPUTCHANNEL, N] = size(this.pI.O);
            this.P = randn([KERNELWIDTHS INPUTCHANNEL OUTPUTCHANNELA*OUTPUTCHANNELB]) * 0.01;
            this.O = zeros([H W D T OUTPUTCHANNELA*OUTPUTCHANNELB N]);
            this.gradP = zeros(size(this.P));
            this.gradO = zeros(size(this.O));

            % 初始化固定核
            % size(this.P)
            gaborArray = util.gaborFilterBank(OUTPUTCHANNELA,OUTPUTCHANNELB,KERNELWIDTHS(1),KERNELWIDTHS(2));
            gaborArray = cell2mat(gaborArray);
            gaborArray = reshape(gaborArray, [KERNELWIDTHS(1) OUTPUTCHANNELA KERNELWIDTHS(2) OUTPUTCHANNELB]);
            gaborArray = permute(gaborArray, [1 3 2 4]);
            gaborArray = real(gaborArray);
            this.P = reshape(gaborArray, [KERNELWIDTHS 1 OUTPUTCHANNELA*OUTPUTCHANNELB]);
            this.P = repmat(this.P, [1 1 1 1 INPUTCHANNEL 1]);
            % size(this.P)
            % for i=1:INPUTCHANNEL
            %     figure
            %     for j=1:OUTPUTCHANNELA
            %         for k=1:OUTPUTCHANNELB
            %             subplot(OUTPUTCHANNELA,OUTPUTCHANNELB,(j-1)*OUTPUTCHANNELB+k);
            %             imshow(this.P(:,:,1,1,i,(k-1)*OUTPUTCHANNELA+j), []);
            %         end
            %     end
            % end
            % drawnow;
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            [H, W, D, T, INPUTCHANNEL, N] = size(this.pI.O);
            OUTPUTCHANNEL = size(this.P, 6);
            
            thispIO = this.pI.O;
            thisP = this.P;
            thisO = zeros([H W D T OUTPUTCHANNEL N]);
            parfor i=1:N
                for j=1:OUTPUTCHANNEL
                    for k=1:INPUTCHANNEL
                        tmp = convn(thispIO(:,:,:,:,k,i), thisP(:,:,:,:,k,j), 'same');
                        thisO(:,:,:,:,j,i) = thisO(:,:,:,:,j,i) + tmp;
                    end
                end
            end
            this.O = thisO;
            
            %toc;
        end
        
        function [] = bp(this)
            warning(util.funname());
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
        end
        
        function [] = setPs(this, Ps)
            this.pI.setPs(Ps);
        end
    end
end

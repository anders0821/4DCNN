classdef Data < handle
    properties
        data;
        MINIBATCHSIZE;
        miniBatchOffset;
        O;
        gradO;
    end
    
    methods
        function [this] = Data(data, MINIBATCHSIZE)
            assert(MINIBATCHSIZE>=1);
            assert(MINIBATCHSIZE<=size(data, 4));
            
            % 初始化成员变量
            this.data = data;
            this.MINIBATCHSIZE = MINIBATCHSIZE;
            this.miniBatchOffset = 1;
            this.O = zeros(size(data,1), size(data,2), size(data,3), MINIBATCHSIZE);
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 计算miniBatchIdx
            N = size(this.data, 4);
            S = this.miniBatchOffset;
            E = this.miniBatchOffset+this.MINIBATCHSIZE-1;
            if(E<=N)
                idx = S:E;
            else
                idx = [S:N 1:E-N];
            end
            %fprintf('Fetch Data');
            %fprintf(' %d', idx);
            %fprintf('. ', idx);
            assert(size(idx,2)==this.MINIBATCHSIZE);
            this.miniBatchOffset = mod(E, N)+1;
            
            % 输出
            this.O = this.data(:,:,:,idx);
            
            %toc;
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            % 不做任何处理
            %toc;
        end
        
        function [total, totalP] = debugMemory(this)
            disp(util.funname());
            
            fprintf('	O');
            fprintf('	%d', size(this.O));
            fprintf('	(%d)', numel(this.O));
            fprintf('\n');
            
            fprintf('	data');
            fprintf('	%d', size(this.data));
            fprintf('	(%d)', numel(this.data));
            fprintf('\n');
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            total = numel(this.O)+numel(this.data)+numel(this.gradO);
            totalP = 0;
        end
        
        function [p] = debugHist(this, fig1, fig2, fig3, fig4, m, n, p)
            %p = this.pI.debugHist(fig1, fig2, fig3, fig4, m, n, p);
            
            p = p+1;
            
            if(~isempty(fig1))
                set(0, 'CurrentFigure', fig1);
                subplot(m,n,p);
                hist(this.O(:),100);
                title(class(this));
            end
            
            if(~isempty(fig3))
                set(0, 'CurrentFigure', fig3);
                subplot(m,n,p);
                hist(this.gradO(:),100);
                title(class(this));
            end
        end
        
        function [Ps] = getPs(this)
            Ps = cell(0);
        end
        
        function [] = setPs(this, Ps)
        end
    end
end

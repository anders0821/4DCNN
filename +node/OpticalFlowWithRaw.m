classdef OpticalFlowWithRaw < handle
    properties
        pI;
        O;
        gradO;
    end
    
    methods
        function [this] = OpticalFlowWithRaw(pI)
            this.pI = pI;
            
            [H, W, D, T, C, N] = size(this.pI.O);
            assert(T==1);
            assert(C==1)
            % Dͨ���ϼ������ ����֡����Ϊ�ڶ�֡���� ��֤Dά�Ȳ���
            % �����ͨ��������������Ĺ���1->3
            this.O = zeros([H W D T 3 N]);
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            
            [H, W, D, T, C, N] = size(this.pI.O);
            assert(T==1);
            assert(C==1);
            for i=1:N
                % ������������
                optical_flow_calculator = vision.OpticalFlow('Method','Horn-Schunck',... % HS����
                    'ReferenceFrameSource', 'Property', 'ReferenceFrameDelay', 1,... % ��֡������� ʹ�����û���
                    'Smoothness', 1,...
                    'IterationTerminationCondition', 'Maximum iteration count', 'MaximumIterationCount', 10, ... % ������ֹ����
                    'OutputValue',  'Horizontal and vertical components in complex form' ... % ���������������
                    );
                
                % ��֡�����
                for j=1:D
                    im = this.pI.O(:,:,j,:,:,i);
                    optical_flow_vector = step(optical_flow_calculator, im);
                    this.O(:,:,j,:,1,i) = real(optical_flow_vector);
                    this.O(:,:,j,:,2,i) = imag(optical_flow_vector);
                end
            end
            
            % ����֡����Ϊ�ڶ�֡���� ��֤Dά�Ȳ���
            this.O(:,:,1,:,:,:) = this.O(:,:,2,:,:,:);
            
            % ���������ͨ��Ϊԭʼ���� �������ֲ�(mean std)�����������ͬ
            this.O(:,:,:,:,3,:) = (this.pI.O - 0.5) / 18.148;
            
            % ���ӻ�this.O
%             OUTPUTCHANNEL = size(this.O, 5);
%             for i=1:N
%                 figure;
%                 for j=1:OUTPUTCHANNEL
%                     for k=1:D
%                         subplot(OUTPUTCHANNEL,D,(j-1)*D+k);
%                         im = this.O(:,:,k,1,j,i);
%                         if(j==3)
%                             imshow(im);
%                         else
%                             imshow(im, [-0.1 0.1]);
%                         end
%                     end
%                 end
%             end
%             
%             figure;
%             a = this.O(:,:,:,:,1:2,:);
%             a = a(:);
%             hist(a,1000);
%             xlim([-0.1 0.1]);
%             [mean(a) std(a)]
%             
%             figure;
%             a = this.O(:,:,:,:,3,:);
%             a = a(:);
%             hist(a,1000);
%             xlim([-0.1 0.1]);
%             [mean(a) std(a)]
%             
%             asd
            
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
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            total = total + numel(this.O)+numel(this.gradO);
            totalP = totalP + 0;
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
            
            if(~isempty(fig3))
                set(0, 'CurrentFigure', fig3);
                subplot(m,n,p);
                hist(this.gradO(:),100);
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

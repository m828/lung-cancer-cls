class RobustMultiModalModel(nn.Module):
    def __init__(self, gene_in_dim, text_dim=768, num_classes=3, target_feat_dim=512):
        super().__init__()
        self.target_feat_dim = target_feat_dim

        self.ct_backbone = monai.networks.nets.DenseNet121(
            spatial_dims=3, n_input_channels=1, out_channels=512
        )
        # self.ct_proj = nn.Linear(target_feat_dim, target_feat_dim)
        self.ct_norm = nn.LayerNorm(target_feat_dim)

        # 文本分支：投影层（将BERT嵌入映射到目标维度）
        self.text_proj = nn.Linear(text_dim, target_feat_dim)
        self.text_norm = nn.LayerNorm(target_feat_dim)

        # 基因分支：MLP + 投影层
        self.gene_mlp = nn.Sequential(
            nn.Linear(gene_in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, target_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.6)
        )
        # self.gene_proj = nn.Linear(target_feat_dim, target_feat_dim)
        self.gene_norm = nn.LayerNorm(target_feat_dim)

        # 门控融合模块（根据模态存在性动态调整权重）
        self.gate_ct = nn.Sequential(nn.Linear(target_feat_dim, 1), nn.Sigmoid())
        self.gate_text = nn.Sequential(nn.Linear(target_feat_dim, 1), nn.Sigmoid())
        self.gate_gene = nn.Sequential(nn.Linear(target_feat_dim, 1), nn.Sigmoid())

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(target_feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, num_classes)
        )

    def forward(self, ct_data, text_emb, gene_feat_input, modal_mask):
        # 初始化特征为零向量
        batch_size = ct_data.size(0)
        ct_feat = torch.zeros((batch_size, self.target_feat_dim), device=ct_data.device)
        text_feat = torch.zeros((batch_size, self.target_feat_dim), device=ct_data.device)
        gene_feat = torch.zeros((batch_size, self.target_feat_dim), device=ct_data.device)

        # 处理CT特征（仅当模态存在时）
        if modal_mask[:, 0].any():
            ct_feat_temp = self.ct_backbone(ct_data)
            # ct_feat = self.ct_proj(ct_feat)
            ct_feat_temp = self.ct_norm(ct_feat_temp)
            ct_feat = ct_feat_temp * modal_mask[:, 0].unsqueeze(1)  # 缺失模态权重置0

        # 处理文本特征（仅当模态存在时）
        if modal_mask[:, 1].any():
            text_feat_temp = self.text_proj(text_emb)
            text_feat_temp = self.text_norm(text_feat_temp)
            text_feat = text_feat_temp * modal_mask[:, 1].unsqueeze(1)

        # 处理基因特征（仅当模态存在时）
        if modal_mask[:, 2].any():
            gene_feat_temp = self.gene_mlp(gene_feat_input)
            # gene_feat = self.gene_proj(gene_feat)
            gene_feat_temp = self.gene_norm(gene_feat_temp)
            gene_feat = gene_feat_temp * modal_mask[:, 2].unsqueeze(1)

        # 计算门控权重
        w_ct = self.gate_ct(ct_feat)
        w_text = self.gate_text(text_feat)
        w_gene = self.gate_gene(gene_feat)

        # 加权融合（缺失模态自动不贡献）
        fused_feat = w_ct * ct_feat + w_text * text_feat + w_gene * gene_feat

        # 分类
        logits = self.classifier(fused_feat)
        return logits
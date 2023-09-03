import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit
import time
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel
from vilt.modules import heads, objectives, vilt_utils
import clip 

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        # self.txt_proj = nn.Linear(768, 768)
        # self.img_proj = nn.Linear(768, 768)
        # self.txt_proj.apply(weights_init_kaiming)
        # self.img_proj.apply(weights_init_kaiming)


        clip_model,_ = clip.load('/root/paddlejob/workspace/env_run/output/shaozhiyin/cache/clip_model/ViT-B-16.pt')
        self.clip = clip_model.float()


        self.text_model = BertModel.from_pretrained('/root/paddlejob/workspace/env_run/output/shaozhiyin/data/bert_base_uncased')
        # for n,p in self.text_model.named_parameters():
        #     if '.11.' not in n and '.10.' not in n :
        #         p.requires_grad = False
        # self.text_model.apply(objectives.init_weights)

        # self.text_embeddings = BertEmbeddings(bert_config)
        # self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)
        feature_length = 768
        self.txt_proj = nn.Linear(768, feature_length)
        self.img_proj = nn.Linear(768, feature_length)
        self.txt_proj.apply(objectives.init_weights)
        self.img_proj.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        # aaa = time.time()
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_ids_nomask = batch[f"text_ids"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # a__ = time.time()
        # text_embeds = self.text_embeddings(text_ids)
        # a_= time.time()
        # print('Txtembed_time:',a_-a__)
        img = batch[imgkey][0]
        # if image_embeds is None and image_masks is None:
        #     img = batch[imgkey][0]
        #     # e_ = time.time()
        #     # print('getimage_time:',e_-a_)
        #     # print(img.shape)
        #     (
        #         image_embeds,
        #         image_masks,
        #         patch_index,
        #         image_labels,
        #     ) = self.transformer.visual_embed(
        #         img,
        #         max_image_len=self.hparams.config["max_image_len"],
        #         mask_it=mask_image,
        #     )
        # else:
        #     patch_index, image_labels = (
        #         None,
        #         None,
        #     )
        # b_ = time.time()
        # print('Imgembed_time:',b_-a_)
        # text_embeds, image_embeds = (
        #     text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
        #     image_embeds
        #     + self.token_type_embeddings(
        #         torch.full_like(image_masks, image_token_type_idx)
        #     ),
        # )
        # print(text_embeds)
        # print(text_ids[0])
        # print(text_ids_nomask[0])
        # print(text_labels[0])
        # print(text_masks)
        # print(image_masks)
        # co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        # co_masks = torch.cat([text_masks, image_masks], dim=1)


        x = text_ids
        # c_ = time.time()
        # for i, blk in enumerate(self.transformer.blocks):
        #     x, _attn = blk(x, mask=text_masks,type = 'image')
        x = self.text_model(x, attention_mask=text_masks)[0]
        # x = self.transformer.norm(x)
        text_feats = x
        text_feature = self.txt_proj(text_feats)
        text_feature,_ = text_feature.max(dim=1)

        x = img

        latent_visual, mask, ids_restore, ids_keep, ids_mask = self.transformer.forward_encoder_cae(x,mask_ratio=0.5)
        latent_mask, latent_visual = self.transformer.forward_decoder_cae(latent_visual,ids_restore,ids_keep,ids_mask)
        target_mask, target_visual = self.forward_clip_cae(x,ids_keep, ids_mask)
        latent_visual = latent_visual[:,1:]
        cae_loss = self.transformer.forward_loss_cae(latent_mask,latent_visual,target_mask,target_visual)
        # pred = self.transformer.forward_decoder_cae(latent_visual, )
        # mae_loss = self.transformer.forward_loss_mae(img, pred, mask)
        
        x = self.transformer.forward_encoder_nomask(img)
        # c_ = time.time()
        # for i, blk in enumerate(self.transformer.blocks):
        #     x, _attn = blk(x, mask=image_masks, type = 'image')
        # x = self.transformer.norm(x)
        image_feats = x
        image_feature = self.img_proj(image_feats)
        image_feature,_ = image_feature.max(dim=1)
        
        co_feats = torch.cat([text_feats,image_feats],dim=1)
        cls_feats = self.pooler(co_feats)
        cls_feats = self.pooler(x)

        # f_ = time.time()
        # print('pooler_time:',f_-d_)
        # print('whole_time:',f_-a__)
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "text_feature": text_feature,
            "image_feature": image_feature,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "mae_loss": cae_loss,
        }
        # bbb = time.time()
        # print('whole_forward:',bbb-aaa)
        return ret

    def forward_clip_cae(self,_x,ids_keep, ids_mask):
        with torch.no_grad():
            target_embed = self.clip.encode_image(_x)
        D = target_embed.size(2)
        target_visual = torch.gather(target_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_mask = torch.gather(target_embed, dim=1, index=ids_mask.unsqueeze(-1).repeat(1, 1, D))

        return target_mask, target_visual

    def forward(self, batch):
        # import time
        # eee = time.time()
        ret = dict()
        if len(self.current_tasks) == 0:
            # print('infer--------------------')
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            # print('mlm--------------------')
            ret.update(objectives.compute_mlm(self, batch))

        if "cl" in self.current_tasks:
            # print('cl--------------------')
            ret.update(objectives.compute_cl(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            # print('mpp--------------------')
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            # print('itm--------------------')
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            # print('vqa--------------------')
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            # print('nlvr2--------------------')
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            # print('irtr--------------------')
            ret.update(objectives.compute_irtr(self, batch))
        # print('model_forward:',time.time()-eee)
        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)

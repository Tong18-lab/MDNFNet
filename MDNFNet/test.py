import time
import datetime
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes

softmax = nn.Softmax(dim=1)

def concat_all_gather(tensors, num_total_examples):
    '''
    Performs all_gather operation on the provided tensor list.
    '''
    outputs = []
    for tensor in tensors:
        tensor = tensor.cuda()
        tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor)
        output = torch.cat(tensors_gather, dim=0).cpu()
        outputs.append(output[:num_total_examples])
    return outputs

@torch.no_grad()
def extract_img_feature(config, model, dataloader):
    avgpool = nn.AdaptiveAvgPool2d(1)
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    for batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
        flip_imgs = torch.flip(imgs, [3])
        imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
        batch_features = model(imgs)
        batch_features = avgpool(batch_features).view(batch_features.size(0), -1)
        batch_features_flip = model(flip_imgs)
        batch_features_flip = avgpool(batch_features_flip).view(batch_features_flip.size(0), -1)
        batch_features += batch_features_flip
        batch_features = F.normalize(batch_features, p=2, dim=1)
        features.append(batch_features.cpu())
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)

    features = torch.cat(features, 0)
    return features, pids, camids, clothes_ids


def test(config, model, attention, gap_classifier, gap_classifier_h, gap_classifier_b, queryloader, galleryloader,
         dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    attention.eval()
    gap_classifier.eval()
    gap_classifier_h.eval()
    gap_classifier_b.eval()
    # Extract features
    qf, q_pids, q_camids, q_clothes_ids = extract_img_feature(config, model, queryloader)
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(config, model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids],
                                                            len(dataset.query))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids],
                                                            len(dataset.gallery))

    torch.cuda.empty_cache()
    time_elapsed = time.time() - since
    logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    since = time.time()
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda()
    # Cosine distance
    for i in range(m):
        distmat[i] = (-torch.mm(qf[i:i + 1], gf.t())).cpu()
    distmat = distmat.numpy()
    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    time_elapsed = time.time() - since

    logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    since = time.time()
    logger.info("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    time_elapsed = time.time() - since
    logger.info('Using {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if config.DATA.DATASET in ['deepchange', 'vcclothes_sc', 'vcclothes_cc']: return cmc[0]

    logger.info("Computing CMC and mAP only for the same clothes setting")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='SC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    logger.info("Computing CMC and mAP only for clothes-changing")
    cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                     mode='CC')
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    return cmc[0]


def test_prcc(config, model, attention, gap_classifier, gap_classifier_h, gap_classifier_b, queryloader_same,
              queryloader_diff, galleryloader, dataset):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    attention.eval()
    gap_classifier.eval()
    gap_classifier_h.eval()
    gap_classifier_b.eval()

    # Extract features for query set
    qsf, qs_pids, qs_camids, qs_clothes_ids = extract_img_feature(config, model, queryloader_same)
    qdf, qd_pids, qd_camids, qd_clothes_ids = extract_img_feature(config, model, queryloader_diff)
    # Extract features for gallery set
    gf, g_pids, g_camids, g_clothes_ids = extract_img_feature(config, model, galleryloader)
    # Gather samples from different GPUs
    torch.cuda.empty_cache()
    qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather([qsf, qs_pids, qs_camids, qs_clothes_ids],
                                                                len(dataset.query_same))
    qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather([qdf, qd_pids, qd_camids, qd_clothes_ids],
                                                                len(dataset.query_diff))
    gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
    time_elapsed = time.time() - since
    logger.info("Extracted features for query set (with same clothes), obtained {} matrix".format(qsf.shape))
    logger.info("Extracted features for query set (with different clothes), obtained {} matrix".format(qdf.shape))
    logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # Compute distance matrix between query and gallery
    m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
    distmat_same = torch.zeros((m, k))
    distmat_diff = torch.zeros((n, k))
    qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
    # Cosine similarity
    for i in range(m):
        distmat_same[i] = (- torch.mm(qsf[i:i + 1], gf.t())).cpu()
    for i in range(n):
        distmat_diff[i] = (- torch.mm(qdf[i:i + 1], gf.t())).cpu()
    distmat_same = distmat_same.numpy()
    distmat_diff = distmat_diff.numpy()
    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    logger.info("Computing CMC and mAP for the same clothes setting")
    cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    logger.info("Computing CMC and mAP only for clothes changing")
    cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")
    return cmc[0]


# import os
# import time
# import datetime
# import logging
# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch import distributed as dist
# from PIL import Image, ImageDraw
# from tools.eval_metrics import evaluate, evaluate_with_clothes
#
# softmax = nn.Softmax(dim=1)
#
# def concat_all_gather(tensors, num_total_examples):
#     outputs = []
#     for tensor in tensors:
#         tensor = tensor.cuda()
#         tensors_gather = [tensor.clone() for _ in range(dist.get_world_size())]
#         dist.all_gather(tensors_gather, tensor)
#         output = torch.cat(tensors_gather, dim=0).cpu()
#         outputs.append(output[:num_total_examples])
#     return outputs
#
# @torch.no_grad()
# def extract_img_feature(config, model, dataloader):
#     avgpool = nn.AdaptiveAvgPool2d(1)
#     features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
#     img_path_list = []
#     for batch_idx, (img_paths, imgs, batch_pids, batch_camids, batch_clothes_ids) in enumerate(dataloader):
#         flip_imgs = torch.flip(imgs, [3])
#         imgs, flip_imgs = imgs.cuda(), flip_imgs.cuda()
#         batch_features = model(imgs)
#         batch_features = avgpool(batch_features).view(batch_features.size(0), -1)
#         batch_features_flip = model(flip_imgs)
#         batch_features_flip = avgpool(batch_features_flip).view(batch_features_flip.size(0), -1)
#         batch_features += batch_features_flip
#         batch_features = F.normalize(batch_features, p=2, dim=1)
#         features.append(batch_features.cpu())
#         pids = torch.cat((pids, batch_pids.cpu()), dim=0)
#         camids = torch.cat((camids, batch_camids.cpu()), dim=0)
#         clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
#         img_path_list.extend(img_paths)
#     features = torch.cat(features, 0)
#     return features, pids, camids, clothes_ids, img_path_list
#
# def visualize_ranked_results(distmat, q_paths, g_paths, q_pids, g_pids, save_dir, topk=10, max_vis=2000,
#                              resize_size=(128, 256)):
#     os.makedirs(save_dir, exist_ok=True)
#     query_dir = os.path.join(save_dir, 'query')
#     gallery_dir = os.path.join(save_dir, 'gallery_topk')
#     os.makedirs(query_dir, exist_ok=True)
#     os.makedirs(gallery_dir, exist_ok=True)
#
#     num_q = distmat.shape[0]
#
#     for q_idx in range(min(num_q, max_vis)):
#         q_img_path = q_paths[q_idx]
#         q_pid = q_pids[q_idx]
#         sorted_indices = np.argsort(distmat[q_idx])
#
#         # 保存查询图（原图或resize）
#         q_img = Image.open(q_img_path).convert('RGB')
#         q_img_resized = q_img.resize(resize_size)
#         q_img_resized.save(os.path.join(query_dir, f"query_{q_idx:03d}.jpg"))
#
#         # 保存Top-K gallery图拼接
#         gallery_imgs = []
#         for rank, g_idx in enumerate(sorted_indices[:topk]):
#             g_img = Image.open(g_paths[g_idx]).convert('RGB')
#             g_pid = g_pids[g_idx]
#
#             # resize
#             g_img = g_img.resize(resize_size)
#             draw = ImageDraw.Draw(g_img)
#             border_color = 'green' if q_pid == g_pid else 'red'
#             draw.rectangle([0, 0, g_img.width - 1, g_img.height - 1], outline=border_color, width=5)
#             gallery_imgs.append(g_img)
#
#         # 横向拼接
#         total_width = resize_size[0] * topk
#         new_img = Image.new('RGB', (total_width, resize_size[1]), (255, 255, 255))
#         for i, im in enumerate(gallery_imgs):
#             new_img.paste(im, (i * resize_size[0], 0))
#
#         new_img.save(os.path.join(gallery_dir, f"query_{q_idx:03d}_top{topk}.jpg"))
#
# def test(config, model, attention, gap_classifier, gap_classifier_h, gap_classifier_b, queryloader, galleryloader, dataset):
#     logger = logging.getLogger('reid.test')
#     since = time.time()
#     model.eval()
#     attention.eval()
#     gap_classifier.eval()
#     gap_classifier_h.eval()
#     gap_classifier_b.eval()
#
#     qf, q_pids, q_camids, q_clothes_ids, q_paths = extract_img_feature(config, model, queryloader)
#     gf, g_pids, g_camids, g_clothes_ids, g_paths = extract_img_feature(config, model, galleryloader)
#
#     torch.cuda.empty_cache()
#     qf, q_pids, q_camids, q_clothes_ids = concat_all_gather([qf, q_pids, q_camids, q_clothes_ids], len(dataset.query))
#     gf, g_pids, g_camids, g_clothes_ids = concat_all_gather([gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
#     # 聚合图像路径（每个进程收集所有）
#     q_paths_all = [None for _ in range(dist.get_world_size())]
#     g_paths_all = [None for _ in range(dist.get_world_size())]
#     dist.all_gather_object(q_paths_all, q_paths)
#     dist.all_gather_object(g_paths_all, g_paths)
#     q_paths = sum(q_paths_all, [])
#     g_paths = sum(g_paths_all, [])
#
#     time_elapsed = time.time() - since
#     logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))
#     logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
#     logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#
#     m, n = qf.size(0), gf.size(0)
#     distmat = torch.zeros((m, n))
#     qf, gf = qf.cuda(), gf.cuda()
#     for i in range(m):
#         distmat[i] = (-torch.mm(qf[i:i + 1], gf.t())).cpu()
#     distmat = distmat.numpy()
#     q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
#     g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
#
#     logger.info("Computing CMC and mAP")
#     cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
#     logger.info("Results ---------------------------------------------------")
#     logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
#     logger.info("-----------------------------------------------------------")
#
#
#     if config.DATA.DATASET in ['deepchange', 'vcclothes_sc', 'vcclothes_cc']:
#         return cmc[0]
#
#     logger.info("Computing CMC and mAP only for the same clothes setting")
#     cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='SC')
#     logger.info("Results ---------------------------------------------------")
#     logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
#     logger.info("-----------------------------------------------------------")
#
#     logger.info("Computing CMC and mAP only for clothes-changing")
#     cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, mode='CC')
#     logger.info("Results ---------------------------------------------------")
#     logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
#     logger.info("-----------------------------------------------------------")
#     # clothes-changing 可视化（仅此保存）
#     logger.info("Saving Top-K retrieval visualizations (clothes-changing only)...")
#     vis_dir_diff = os.path.join(config.OUTPUT, "topk_vis_diff")
#     visualize_ranked_results(distmat, q_paths, g_paths, q_pids, g_pids, vis_dir_diff, topk=10)
#
#     return cmc[0]
# def test_prcc(config, model, attention, gap_classifier, gap_classifier_h, gap_classifier_b, queryloader_same,
#               queryloader_diff, galleryloader, dataset):
#     logger = logging.getLogger('reid.test')
#     since = time.time()
#     model.eval()
#     attention.eval()
#     gap_classifier.eval()
#     gap_classifier_h.eval()
#     gap_classifier_b.eval()
#
#     # Extract features
#     qsf, qs_pids, qs_camids, qs_clothes_ids, qs_paths = extract_img_feature(config, model, queryloader_same)
#     qdf, qd_pids, qd_camids, qd_clothes_ids, qd_paths = extract_img_feature(config, model, queryloader_diff)
#     gf, g_pids, g_camids, g_clothes_ids, g_paths = extract_img_feature(config, model, galleryloader)
#
#     torch.cuda.empty_cache()
#
#     # 多GPU同步特征
#     qsf, qs_pids, qs_camids, qs_clothes_ids = concat_all_gather(
#         [qsf, qs_pids, qs_camids, qs_clothes_ids], len(dataset.query_same))
#     qdf, qd_pids, qd_camids, qd_clothes_ids = concat_all_gather(
#         [qdf, qd_pids, qd_camids, qd_clothes_ids], len(dataset.query_diff))
#     gf, g_pids, g_camids, g_clothes_ids = concat_all_gather(
#         [gf, g_pids, g_camids, g_clothes_ids], len(dataset.gallery))
#
#     # 同步路径
#     qs_paths_all = [None for _ in range(dist.get_world_size())]
#     qd_paths_all = [None for _ in range(dist.get_world_size())]
#     g_paths_all = [None for _ in range(dist.get_world_size())]
#     dist.all_gather_object(qs_paths_all, qs_paths)
#     dist.all_gather_object(qd_paths_all, qd_paths)
#     dist.all_gather_object(g_paths_all, g_paths)
#     qs_paths = sum(qs_paths_all, [])
#     qd_paths = sum(qd_paths_all, [])
#     g_paths = sum(g_paths_all, [])
#
#     time_elapsed = time.time() - since
#     logger.info("Extracted features for query set (same clothes), obtained {} matrix".format(qsf.shape))
#     logger.info("Extracted features for query set (diff clothes), obtained {} matrix".format(qdf.shape))
#     logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
#     logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#
#     # 计算距离矩阵
#     m, n, k = qsf.size(0), qdf.size(0), gf.size(0)
#     distmat_same = torch.zeros((m, k))
#     distmat_diff = torch.zeros((n, k))
#     qsf, qdf, gf = qsf.cuda(), qdf.cuda(), gf.cuda()
#
#     for i in range(m):
#         distmat_same[i] = (- torch.mm(qsf[i:i + 1], gf.t())).cpu()
#     for i in range(n):
#         distmat_diff[i] = (- torch.mm(qdf[i:i + 1], gf.t())).cpu()
#
#     distmat_same = distmat_same.numpy()
#     distmat_diff = distmat_diff.numpy()
#     qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
#     qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
#     g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
#
#     # same clothes 评估
#     logger.info("Computing CMC and mAP for the same clothes setting")
#     cmc, mAP = evaluate(distmat_same, qs_pids, g_pids, qs_camids, g_camids)
#     logger.info("Results ---------------------------------------------------")
#     logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
#         cmc[0], cmc[4], cmc[9], cmc[19], mAP))
#     logger.info("-----------------------------------------------------------")
#
#
#
#     # diff clothes 评估
#     logger.info("Computing CMC and mAP for the clothes-changing setting")
#     cmc, mAP = evaluate(distmat_diff, qd_pids, g_pids, qd_camids, g_camids)
#     logger.info("Results ---------------------------------------------------")
#     logger.info('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(
#         cmc[0], cmc[4], cmc[9], cmc[19], mAP))
#     logger.info("-----------------------------------------------------------")
#
#     # 可视化 diff clothes
#     logger.info("Saving Top-K retrieval visualizations (clothes-changing)...")
#     vis_dir_diff = os.path.join(config.OUTPUT, "topk_vis_diff")
#     visualize_ranked_results(distmat_diff, qd_paths, g_paths, qd_pids, g_pids, vis_dir_diff, topk=10)
#
#     return cmc[0]

import torch 

def pairtrain_epoch(model, train_loader, valid_loader, optimizer, device, _alpha=10):
    '''
    train a network with LD-PA paradgim. 
    '''
    ###################### Train ######################
    model.train()
    loss_cls_sum = 0
    loss_ld_sum = 0
    # loss_sum = 0
    train_batch = 0
    for (xtar, ytar), (fref, _) in train_loader:
        xtar, ytar, fref = xtar.to(device), ytar.to(device), fref.to(device)

        optimizer.zero_grad() # clear gradient
        ftar, ytar_pred = model(xtar)
        loss_cls = model.clsloss(ytar_pred, ytar) # classification loss
        loss_ld = model.ldloss(ftar, fref)        # leakage distillation loss
        loss = loss_cls + _alpha*loss_ld

        loss.backward()   # backward propagation
        optimizer.step()  # update parameter
        loss_cls_sum += loss_cls.item()
        loss_ld_sum += loss_ld.item()
        # loss_sum += loss.item()
        train_batch += 1

    ###################### Validation ######################
    model.eval()
    loss_eval = 0
    valid_batch = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            _, y_pred = model(x)
            loss_eval += model.clsloss(y_pred, y).item()
            valid_batch += 1

    return {'train_cls_loss': loss_cls_sum/train_batch, 
            'train_ld_loss': loss_ld_sum/train_batch, 
            # 'train_loss': loss_sum/train_batch,
            'valid_cls_loss': loss_eval/valid_batch}


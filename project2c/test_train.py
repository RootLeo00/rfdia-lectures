def train(epochs=20, lr=0.6, alpha=10, beta=0.75, activation='sigmoid'):
    dann = DANN(activation).cuda()

    optimizer = torch.optim.SGD(dann.parameters(), lr=lr, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: 0.01 / (1 + alpha * e / epochs) ** beta
    )
    lmbd = lambda e: -1 + 2 / (1 + math.exp(-2 * e / (len(source_train_loader) * epochs)))

    b = 0
    for epoch in range(epochs):
        cls_loss, domain_loss = 0., 0.
        grl_factor = lmbd(b)
        print(f"GRL factor {grl_factor}" )

        for (xs, ys), (xt, _) in zip(source_train_loader, target_train_loader):
            grl_factor = lmbd(b)
            b += 1

            batch_size = xs.shape[0] # 64

            xs, ys = xs.cuda(), ys.cuda()
            xt = xt.cuda()
            x = torch.cat((xs, xt), dim=0)

            optimizer.zero_grad()
            cls_logits, domain_logits = dann(x, factor=-grl_factor)

            yhat_s, yhat_t = cls_logits.chunk(2, dim=0)
            # xhat_s, xhat_t = domain_logits.chunk(2, dim=0)

            domain_logits_labels = torch.cat((torch.ones(batch_size, 1), torch.zeros(batch_size, 1)), dim=0).cuda()

            ce = nn.CrossEntropyLoss()  # For classification loss TODO
            ce = ce(yhat_s, ys)
            bce = nn.BCEWithLogitsLoss()  # For binary cross-entropy loss TODO
            bce = bce(domain_logits, domain_logits_labels) # [64 -> 1, 64 -> 0]
            loss = ce + bce * 0.6
            loss.backward()
            optimizer.step()

            cls_loss += ce.item()
            domain_loss += bce.item()

        cls_loss = round(cls_loss / len(source_train_loader), 5)
        domain_loss = round(domain_loss / (2 * len(source_train_loader)), 5)
        print(f'Epoch {epoch}, class loss: {cls_loss}, domain loss: {domain_loss}')
        scheduler.step()

    c_acc, d_acc, c_loss, d_loss = eval_dann(dann, source_test_loader)
    print(f"[SOURCE] Class loss/acc: {c_loss} / {c_acc}%, Domain loss/acc: {d_loss} / {d_acc}%")

    c_acc, d_acc, c_loss, d_loss = eval_dann(dann, target_test_loader, source=False)
    print(f"[TARGET] Class loss/acc: {c_loss} / {c_acc}%, Domain loss/acc: {d_loss} / {d_acc}%")

    source_emb = extract_emb(dann, source_train_loader)
    target_emb = extract_emb(dann, target_train_loader)

    print("Original embeddings of source / target", source_emb.shape, target_emb.shape)

    indexes = np.random.permutation(len(source_emb))[:1000]

    emb = np.concatenate((source_emb[indexes], target_emb[indexes]))
    domains = np.concatenate((np.ones((1000,)), np.zeros((1000,))))

    print("Samples embeddings", emb.shape, domains.shape)

    tsne = TSNE(n_components=2)

    emb_2d = tsne.fit_transform(emb)
    print("Dimension reduced embeddings", emb_2d.shape)

    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=domains)
    plt.title("With domain adaptation")
    plt.show()
    
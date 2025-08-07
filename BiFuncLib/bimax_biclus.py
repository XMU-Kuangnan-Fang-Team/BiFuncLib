import pandas as pd

from BiFuncLib.BiclustResult import BiclustResult


def apriori_bimax(matrix, minr=2, minc=2, number=100):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    row_masks = []
    for row in matrix:
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= 1 << j
        row_masks.append(m)
    L1 = []
    for j in range(cols):
        bit = 1 << j
        sup = sum(1 for rm in row_masks if (rm & bit) == bit)
        if sup >= minr:
            L1.append(((j,), bit, sup))
    freq_sets = []
    if minc <= 1:
        freq_sets.extend(L1)
    k = 1
    prev_L = L1
    while prev_L and k < cols:
        Ck = {}
        n = len(prev_L)
        for a in range(n):
            items_a, mask_a, _ = prev_L[a]
            for b in range(a+1, n):
                items_b, mask_b, _ = prev_L[b]
                if items_a[:k-1] == items_b[:k-1]:
                    new_items = tuple(sorted(set(items_a) | set(items_b)))
                    if len(new_items) != k+1:
                        continue
                    new_mask = 0
                    for c in new_items:
                        new_mask |= 1 << c
                    if new_items in Ck:
                        continue
                    sup = sum(1 for rm in row_masks if (rm & new_mask) == new_mask)
                    if sup >= minr:
                        Ck[new_items] = (new_items, new_mask, sup)
        k += 1
        prev_L = list(Ck.values())
        if k >= minc:
            freq_sets.extend(prev_L)
    freq_sets.sort(key=lambda x: (len(x[0]), x[2]), reverse=True)
    maximal = []
    for items, mask, sup in freq_sets:
        s_items = set(items)
        if not any(s_items.issubset(set(other[0])) for other in maximal):
            maximal.append((items, mask, sup))
        if len(maximal) >= number:
            break
    biclusters = []
    for items, mask, sup in maximal:
        rows_res = [i for i, rm in enumerate(row_masks) if (rm & mask) == mask]
        cols_res = list(items)
        biclusters.append({'rows': rows_res, 'cols': cols_res})
    return biclusters


def bimax_biclus(matrix, minr=2, minc=2, number=100):
    raw = apriori_bimax(matrix, minr, minc, number)
    bic_n = len(raw)
    R = len(matrix)
    C = len(matrix[0]) if R else 0
    RowxNumber = [[False] * bic_n for _ in range(R)]
    NumberxCol = [[False] * C     for _ in range(bic_n)]
    for idx, bc in enumerate(raw):
        for r in bc['rows']:
            RowxNumber[r][idx] = True
        for c in bc['cols']:
            NumberxCol[idx][c] = True
    return BiclustResult({
            'Algorithm':     'Apriori-Bimax',
            'minr':          minr,
            'minc':          minc,
            'MaxBiclusters': number
        }, pd.DataFrame(RowxNumber), pd.DataFrame(NumberxCol).T, bic_n, {})


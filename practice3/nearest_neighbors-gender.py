# Plot height vs weight instances from test set
for idx, y in enumerate(y_test):
    # print('idx: {idx}, y: {y}'.format(idx=idx, y=y))
    gender = 'ro' if (y) else 'bo'
    plt.plot(x_test[idx][0], x_test[idx][1], gender)

plt.xlabel('height')
plt.ylabel('weight')
plt.title('Actual Height vs Weight')

plt.savefig('plot_actual_gender_weight_height.png',
            dpi=300, bbox_inches='tight')
plt.clf()
